"""Companion utilities and CLI for the MACE fine-tuning tutorial."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase.io import read, write

from hpc_common import choose_device


FOUNDATIONS = {
    "mp0": {"model": "small", "head": None},
    "mh1": {"model": "mh-1", "head": "omat_pbe"},
}


@dataclass(frozen=True)
class TutorialPaths:
    root: Path
    source_data: Path
    data: Path
    configs: Path
    models: Path
    logs: Path
    results: Path

    @classmethod
    def from_root(cls, root: Path) -> "TutorialPaths":
        root = Path(root)
        work = root / "finetuning"
        return cls(
            root=root,
            source_data=root / "data",
            data=work / "data",
            configs=work / "config",
            models=work / "models",
            logs=work / "logs",
            results=work / "results",
        )

    @property
    def isolated_e0s(self) -> Path:
        return self.data / "isolated_e0s.json"

    @property
    def train_file(self) -> Path:
        return self.data / "solvent_xtb_train_50.xyz"

    @property
    def valid_file(self) -> Path:
        return self.data / "solvent_xtb_valid_50.xyz"

    @property
    def test_file(self) -> Path:
        return self.data / "solvent_xtb_test_1000.xyz"

    def create_output_dirs(self) -> None:
        for directory in (
            self.data,
            self.configs,
            self.models,
            self.logs,
            self.results,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def prepare_finetuning_data(paths: TutorialPaths) -> None:
    """Create deterministic molecular splits and record isolated-atom energies."""
    paths.create_output_dirs()
    source = paths.source_data / "solvent_xtb.xyz"
    if not source.exists():
        raise FileNotFoundError(f"Missing tutorial dataset: {source}")

    configurations = read(source, index=":")
    isolated = [
        atoms
        for atoms in configurations
        if atoms.info.get("config_type") == "IsolatedAtom"
    ]
    molecular = [
        atoms
        for atoms in configurations
        if atoms.info.get("config_type") != "IsolatedAtom"
    ]
    if len(molecular) < 1100:
        raise ValueError(
            f"Expected at least 1100 molecular configurations in {source}, "
            f"found {len(molecular)}"
        )

    write(paths.train_file, molecular[:50])
    write(paths.valid_file, molecular[50:100])
    write(paths.test_file, molecular[100:1100])

    e0s = {
        atoms.get_chemical_symbols()[0]: float(atoms.info["energy_xtb"])
        for atoms in isolated
    }
    paths.isolated_e0s.write_text(
        json.dumps(e0s, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def make_finetune_config(
    paths: TutorialPaths,
    foundation: str,
    device: str,
    epochs: int = 30,
) -> dict[str, Any]:
    spec = FOUNDATIONS[foundation]
    config: dict[str, Any] = {
        "name": f"xtb_{foundation}",
        "foundation_model": spec["model"],
        "train_file": str(paths.train_file),
        "valid_file": str(paths.valid_file),
        "test_file": str(paths.test_file),
        "energy_key": "energy_xtb",
        "forces_key": "forces_xtb",
        "E0s": "estimated",
        "model_dir": str(paths.models),
        "log_dir": str(paths.logs),
        "checkpoints_dir": str(paths.models),
        "results_dir": str(paths.results),
        "device": device,
        "default_dtype": "float32",
        "batch_size": 10,
        "max_num_epochs": epochs,
        "energy_weight": 1.0,
        "forces_weight": 1.0,
        "ema": True,
        "swa": True,
        "seed": 123,
    }
    if spec["head"]:
        config["foundation_head"] = spec["head"]
    return config


def make_replay_config(
    paths: TutorialPaths,
    device: str,
    epochs: int = 20,
    replay_samples: int = 100,
) -> dict[str, Any]:
    config = make_finetune_config(paths, "mh1", device, epochs)
    config.update(
        {
            "name": "xtb_mh1_replay",
            "multiheads_finetuning": True,
            "pseudolabel_replay": True,
            "num_samples_pt": replay_samples,
            "weight_pt_head": 1.0,
        }
    )
    return config


def _run_config(paths: TutorialPaths, name: str, config: dict[str, Any]) -> None:
    executable = shutil.which("mace_run_train")
    if executable is None:
        raise RuntimeError("mace_run_train is not available in this environment")

    config_path = paths.configs / f"{name}.yml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    subprocess.run([executable, "--config", str(config_path)], check=True)


def train_single_head_finetune(
    paths: TutorialPaths,
    foundation: str,
    device: str,
    epochs: int = 30,
) -> None:
    prepare_finetuning_data(paths)
    _run_config(
        paths,
        f"{foundation}_finetune",
        make_finetune_config(paths, foundation, device, epochs),
    )


def train_replay_finetune(
    paths: TutorialPaths,
    device: str,
    epochs: int = 20,
    replay_samples: int = 100,
) -> None:
    prepare_finetuning_data(paths)
    _run_config(
        paths,
        "mh1_replay",
        make_replay_config(paths, device, epochs, replay_samples),
    )


def analyze_e0_reestimation(paths: TutorialPaths, device: str) -> None:
    """Report composition-only corrections for each tutorial foundation."""
    from mace.calculators import mace_mp

    configurations = read(paths.train_file, index=":")
    elements = sorted({symbol for atoms in configurations for symbol in atoms.symbols})
    composition = np.array(
        [[atoms.symbols.count(symbol) for symbol in elements] for atoms in configurations],
        dtype=float,
    )
    targets = np.array(
        [float(atoms.info["energy_xtb"]) for atoms in configurations]
    )

    for name, spec in FOUNDATIONS.items():
        kwargs = {"model": spec["model"], "device": device}
        if spec["head"]:
            kwargs["head"] = spec["head"]
        calculator = mace_mp(**kwargs)
        predictions = []
        for atoms in configurations:
            sample = atoms.copy()
            sample.calc = calculator
            predictions.append(sample.get_potential_energy())
        correction, *_ = np.linalg.lstsq(
            composition,
            targets - np.asarray(predictions),
            rcond=None,
        )
        print(name, dict(zip(elements, correction, strict=True)))


def compare_finetuned_models(paths: TutorialPaths, device: str) -> None:
    """Print energy and force RMSE for fine-tuned models that are present."""
    from mace.calculators import MACECalculator

    configurations = read(paths.test_file, index=":")
    for model_path in sorted(paths.models.glob("*.model")):
        calculator = MACECalculator(
            model_paths=str(model_path),
            device=device,
            default_dtype="float32",
        )
        energy_errors = []
        force_errors = []
        for atoms in configurations:
            sample = atoms.copy()
            sample.calc = calculator
            energy_errors.append(
                (sample.get_potential_energy() - atoms.info["energy_xtb"]) / len(atoms)
            )
            force_errors.extend(
                (sample.get_forces() - atoms.arrays["forces_xtb"]).reshape(-1)
            )
        print(
            model_path.name,
            {
                "energy_rmse_eV_per_atom": float(
                    np.sqrt(np.mean(np.square(energy_errors)))
                ),
                "force_rmse_eV_per_A": float(
                    np.sqrt(np.mean(np.square(force_errors)))
                ),
            },
        )


def write_conditioning_examples(paths: TutorialPaths) -> None:
    """Write input-only charge/spin examples and a matching training template."""
    if not paths.train_file.exists():
        prepare_finetuning_data(paths)

    source = read(paths.train_file, index=0)
    examples = []
    for charge, spin in [(-1, 2), (0, 1), (1, 2)]:
        atoms = source.copy()
        atoms.calc = None
        atoms.info.pop("energy_xtb", None)
        atoms.arrays.pop("forces_xtb", None)
        atoms.info.update(
            total_charge=charge,
            total_spin=spin,
            elec_temp=300.0,
        )
        examples.append(atoms)
    write(paths.data / "conditioning_inputs.xyz", examples)

    template = make_finetune_config(paths, "mh1", "cuda", epochs=30)
    template["embedding_specs"] = {
        "total_charge": {
            "type": "categorical",
            "per": "graph",
            "num_classes": 201,
            "emb_dim": 64,
        },
        "total_spin": {
            "type": "categorical",
            "per": "graph",
            "num_classes": 101,
            "emb_dim": 64,
        },
        "elec_temp": {
            "type": "continuous",
            "per": "graph",
            "in_dim": 1,
            "emb_dim": 32,
        },
    }
    (paths.configs / "conditional_mace_template.yml").write_text(
        yaml.safe_dump(template, sort_keys=False),
        encoding="utf-8",
    )


def run_omol_charge_example(paths: TutorialPaths, device: str) -> None:
    from mace.calculators import mace_omol

    if not paths.train_file.exists():
        prepare_finetuning_data(paths)
    source = read(paths.train_file, index=0)
    calculator = mace_omol(device=device)
    for charge, spin in [(-1, 2), (0, 1), (1, 2)]:
        atoms = source.copy()
        atoms.info.update(charge=charge, spin=spin)
        atoms.calc = calculator
        print(charge, spin, atoms.get_potential_energy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("Tutorials"))
    parser.add_argument(
        "--stage",
        choices=["prepare", "e0s", "mp0", "mh1", "compare", "replay", "properties", "omol"],
        default="prepare",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--replay-samples", type=int, default=100)
    args = parser.parse_args()

    paths = TutorialPaths.from_root(args.work_dir.resolve())
    paths.create_output_dirs()
    device = choose_device(args.device)

    if args.stage == "prepare":
        prepare_finetuning_data(paths)
    elif args.stage == "e0s":
        prepare_finetuning_data(paths)
        analyze_e0_reestimation(paths, device)
    elif args.stage in {"mp0", "mh1"}:
        train_single_head_finetune(paths, args.stage, device, args.epochs)
    elif args.stage == "compare":
        compare_finetuned_models(paths, device)
    elif args.stage == "replay":
        train_replay_finetune(paths, device, args.epochs, args.replay_samples)
    elif args.stage == "properties":
        write_conditioning_examples(paths)
    elif args.stage == "omol":
        run_omol_charge_example(paths, device)


if __name__ == "__main__":
    main()
