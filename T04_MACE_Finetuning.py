#!/usr/bin/env python3
"""Fine-tuning MACE foundation models on the carbonate XTB dataset.

This is the HPC companion to ``T04_MACE_Finetuning.ipynb``.  It compares
single-head fine-tuning from MACE-MP-0 and MACE-MH-1, makes the E0 correction
explicit, demonstrates multihead pseudolabel replay, and finishes with
charge/spin/other-property conditioning examples.

Recommended stage order
-----------------------
``prepare -> e0s -> mp0 -> mh1 -> compare -> replay -> properties``

The optional ``omol`` stage downloads the much larger charge/spin-aware
MACE-OMOL model and evaluates one geometry at several electronic states.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.data import chemical_symbols
from ase.io import read, write
from hpc_common import (
    choose_device,
    configure_logging,
    evaluate,
    find_model,
    print_section,
    require_files,
    set_project_caches,
    train,
    write_yaml,
)

from mace.calculators import MACECalculator, mace_mp, mace_omol

# %% [markdown]
# # Tutorial choices
# Fifty target structures keep the comparison affordable.  The validation set
# is disjoint, and the final 1,000 configurations remain held out for testing.

N_TARGET_TRAIN = 50
N_TARGET_VALID = 50
N_TARGET_TEST = 1000
ELEMENTS = (1, 6, 8)  # H, C, O


@dataclass(frozen=True)
class FoundationSpec:
    """A foundation checkpoint and the head used as the starting point."""

    label: str
    alias: str
    head: str | None
    run_name: str
    seed: int


FOUNDATIONS = {
    "mp0": FoundationSpec(
        label="MACE-MP-0 (small)",
        alias="small",
        head=None,
        run_name="ft_mp0_xtb",
        seed=11,
    ),
    "mh1": FoundationSpec(
        label="MACE-MH-1 (OMAT/PBE head)",
        alias="mh-1",
        head="omat_pbe",
        run_name="ft_mh1_xtb",
        seed=22,
    ),
    "replay": FoundationSpec(
        label="MACE-MH-1 with replay",
        alias="mh-1",
        head="omat_pbe",
        run_name="ft_mh1_xtb_replay",
        seed=33,
    ),
}


@dataclass(frozen=True)
class TutorialPaths:
    """Files shared between notebook cells and restartable HPC stages."""

    root: Path
    data: Path
    configs: Path
    models: Path
    evaluations: Path
    plots: Path
    train: Path
    valid: Path
    test: Path
    isolated_e0s: Path
    e0_analysis: Path

    @classmethod
    def from_root(cls, root: Path) -> "TutorialPaths":
        data = root / "data"
        output = root / "finetuning"
        return cls(
            root=root,
            data=data,
            configs=output / "configs",
            models=output / "models",
            evaluations=output / "evaluations",
            plots=output / "plots",
            train=data / "finetune_xtb_train_50.xyz",
            valid=data / "finetune_xtb_valid_50.xyz",
            test=data / "finetune_xtb_test.xyz",
            isolated_e0s=output / "xtb_isolated_e0s.json",
            e0_analysis=output / "e0_analysis.json",
        )

    def create_output_dirs(self) -> None:
        for directory in (
            self.configs,
            self.models,
            self.evaluations,
            self.plots,
        ):
            directory.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# # Command-line interface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "e0s",
            "mp0",
            "mh1",
            "compare",
            "replay",
            "properties",
            "omol",
        ),
        default="prepare",
    )
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--replay-samples", type=int, default=100)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--enable-cueq", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# %% [markdown]
# ## 1. Prepare a target-level XTB dataset
# Isolated atoms are saved for interpretation, but deliberately excluded from
# the fine-tuning XYZ.  This ensures ``E0s: estimated`` is actually used rather
# than MACE taking explicit isolated-atom energies from the file.


def prepare_finetuning_data(paths: TutorialPaths) -> None:
    """Create molecular-only target splits and record the explicit XTB E0s."""
    print_section("1", "Prepare molecular-only XTB fine-tuning data")
    require_files(paths.root, "data/solvent_xtb.xyz")
    all_configs = read(paths.data / "solvent_xtb.xyz", index=":")

    isolated = []
    molecular = []
    for atoms in all_configs:
        if atoms.info.get("config_type") == "IsolatedAtom" or len(atoms) == 1:
            isolated.append(atoms)
        else:
            molecular.append(atoms)
    required = N_TARGET_TRAIN + N_TARGET_VALID + N_TARGET_TEST
    if len(molecular) < required:
        raise RuntimeError(
            f"Need {required} molecular structures, found {len(molecular)}"
        )

    training = molecular[:N_TARGET_TRAIN]
    validation = molecular[N_TARGET_TRAIN : N_TARGET_TRAIN + N_TARGET_VALID]
    test = molecular[-N_TARGET_TEST:]
    write(paths.train, training)
    write(paths.valid, validation)
    write(paths.test, test)

    xtb_e0s = {
        str(int(atoms.numbers[0])): float(atoms.info["energy_xtb"])
        for atoms in isolated
        if len(atoms) == 1 and "energy_xtb" in atoms.info
    }
    paths.isolated_e0s.parent.mkdir(parents=True, exist_ok=True)
    paths.isolated_e0s.write_text(json.dumps(xtb_e0s, indent=2) + "\n")

    print(f"Training:   {len(training):4d} molecular structures -> {paths.train}")
    print(f"Validation: {len(validation):4d} molecular structures -> {paths.valid}")
    print(f"Test:       {len(test):4d} molecular structures -> {paths.test}")
    print(f"Explicit XTB isolated-atom E0s -> {paths.isolated_e0s}")
    for atomic_number, energy in sorted(xtb_e0s.items(), key=lambda item: int(item[0])):
        print(f"  {chemical_symbols[int(atomic_number)]:>2s}: {energy:12.6f} eV")


# %% [markdown]
# ## 2. Why E0s must change
# MACE decomposes total energy into a composition-dependent reference plus a
# learned environment contribution:
#
#     E(R, Z) = sum_i E0[Z_i] + E_interactions(R, Z)
#
# A foundation model and XTB use different electronic-structure references.
# Their total-energy offset grows with composition and can overwhelm the much
# smaller geometry-dependent signal.  MACE's ``E0s: estimated`` predicts each
# target structure with the foundation model and solves
#
#     A delta_E0 ~= E_XTB - E_foundation
#
# where A contains element counts.  The corrected E0s are the foundation E0s
# plus the least-squares corrections.  This is not the same as fitting average
# atomic energies from scratch: it preserves the foundation model decomposition.


def load_foundation_calculator(
    spec: FoundationSpec, device: str, enable_cueq: bool = False
) -> MACECalculator:
    """Load one foundation model through the public MACE ASE interface."""
    kwargs: dict[str, object] = {
        "model": spec.alias,
        "device": device,
        "default_dtype": "float32",
        "enable_cueq": enable_cueq,
    }
    if spec.head is not None:
        kwargs["head"] = spec.head
    return mace_mp(**kwargs)


def extract_foundation_e0s(
    calculator: MACECalculator, requested_elements: tuple[int, ...] = ELEMENTS
) -> dict[int, float]:
    """Read the selected head's E0 vector from a foundation checkpoint."""
    model = calculator.models[0]
    values = np.asarray(
        model.atomic_energies_fn.atomic_energies.detach().cpu(), dtype=float
    )
    heads = list(getattr(model, "heads", ["Default"]))
    selected_head = calculator.head

    if values.ndim == 1:
        selected = values
    elif values.shape[0] == len(heads):
        selected = values[heads.index(selected_head)]
    elif values.shape[1] == len(heads):
        selected = values[:, heads.index(selected_head)]
    else:
        raise ValueError(f"Cannot interpret foundation E0 shape {values.shape}")

    atomic_numbers = [
        int(number) for number in model.atomic_numbers.detach().cpu().tolist()
    ]
    e0_by_element = dict(zip(atomic_numbers, np.asarray(selected).ravel()))
    return {number: float(e0_by_element[number]) for number in requested_elements}


def estimate_e0_corrections(
    configs: list,
    predicted_energies: np.ndarray,
    foundation_e0s: dict[int, float],
) -> dict[str, object]:
    """Transparent NumPy version of MACE's ``E0s: estimated`` calculation."""
    composition = np.array(
        [
            [np.count_nonzero(atoms.numbers == element) for element in ELEMENTS]
            for atoms in configs
        ],
        dtype=float,
    )
    reference = np.array([atoms.info["energy_xtb"] for atoms in configs])
    error_before = reference - predicted_energies
    corrections, residuals, rank, _ = np.linalg.lstsq(
        composition, error_before, rcond=None
    )
    error_after = error_before - composition @ corrections

    estimated = {
        element: foundation_e0s[element] + float(correction)
        for element, correction in zip(ELEMENTS, corrections)
    }
    atom_counts = np.array([len(atoms) for atoms in configs])
    return {
        "foundation_e0s": foundation_e0s,
        "corrections": dict(zip(ELEMENTS, map(float, corrections))),
        "estimated_e0s": estimated,
        "rank": int(rank),
        "least_squares_residuals": residuals.tolist(),
        "error_before": error_before,
        "error_after": error_after,
        "error_per_atom_before": error_before / atom_counts,
        "error_per_atom_after": error_after / atom_counts,
    }


def analyze_e0_reestimation(
    paths: TutorialPaths, device: str, enable_cueq: bool = False
) -> None:
    """Compare MP-0 and MH-1 energy offsets before and after E0 correction."""
    print_section("2", "Re-estimate E0s for MP-0 and MH-1")
    require_files(paths.root, str(paths.train.relative_to(paths.root)))
    configs = read(paths.train, index=":")
    analysis: dict[str, dict[str, object]] = {}

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, key in zip(axes, ("mp0", "mh1")):
        spec = FOUNDATIONS[key]
        print(f"\nLoading {spec.label}")
        calculator = load_foundation_calculator(spec, device, enable_cueq)
        predictions = []
        for atoms in configs:
            atoms.calc = calculator
            predictions.append(atoms.get_potential_energy())
            atoms.calc = None

        result = estimate_e0_corrections(
            configs,
            np.asarray(predictions),
            extract_foundation_e0s(calculator),
        )
        before = np.asarray(result.pop("error_before"))
        after = np.asarray(result.pop("error_after"))
        before_per_atom = np.asarray(result.pop("error_per_atom_before"))
        after_per_atom = np.asarray(result.pop("error_per_atom_after"))
        result["rmse_per_atom_before_e0"] = float(np.sqrt(np.mean(before_per_atom**2)))
        result["rmse_per_atom_after_e0"] = float(np.sqrt(np.mean(after_per_atom**2)))
        result["total_energy_rmse_before_e0"] = float(np.sqrt(np.mean(before**2)))
        result["total_energy_rmse_after_e0"] = float(np.sqrt(np.mean(after**2)))
        analysis[key] = result

        axis.scatter(before_per_atom, after_per_atom, s=22, alpha=0.7)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set(
            xlabel="Before correction (eV/atom)",
            ylabel="After correction (eV/atom)",
            title=spec.label,
        )

        print("element  foundation E0    correction    estimated XTB-level E0")
        for element in ELEMENTS:
            symbol = chemical_symbols[element]
            foundation = result["foundation_e0s"][element]
            correction = result["corrections"][element]
            estimated = result["estimated_e0s"][element]
            print(
                f"{symbol:>3s} {foundation:14.6f} {correction:13.6f} {estimated:22.6f}"
            )
        print(
            "RMSE of composition offset: "
            f"{result['rmse_per_atom_before_e0']:.4f} -> "
            f"{result['rmse_per_atom_after_e0']:.4f} eV/atom"
        )

    figure.suptitle("Effect of least-squares E0 re-estimation")
    figure.tight_layout()
    plot_path = paths.plots / "e0_reestimation.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    paths.e0_analysis.write_text(json.dumps(analysis, indent=2) + "\n")
    print(f"\nE0 analysis -> {paths.e0_analysis}")
    print(f"E0 plot     -> {plot_path}")


# %% [markdown]
# ## 3. Naive/single-head fine-tuning: MP-0 versus MH-1
# Both runs see identical target data and optimization settings.  They are not
# equal-size architectures; the point is to compare two generations of starting
# knowledge.  ``E0s: estimated`` invokes the same correction introduced above.


def make_finetune_config(
    paths: TutorialPaths,
    foundation_key: str,
    device: str,
    epochs: int,
    enable_cueq: bool = False,
) -> dict[str, object]:
    """Construct a current MACE 0.3.16 single-head fine-tuning configuration."""
    spec = FOUNDATIONS[foundation_key]
    config: dict[str, object] = {
        "model": "MACE",
        "foundation_model": spec.alias,
        "multiheads_finetuning": False,
        "E0s": "estimated",
        "name": spec.run_name,
        "seed": spec.seed,
        "train_file": str(paths.train),
        "valid_file": str(paths.valid),
        "test_file": str(paths.test),
        "energy_key": "energy_xtb",
        "forces_key": "forces_xtb",
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 0.0,
        "lr": 0.001,
        "batch_size": 5,
        "max_num_epochs": epochs,
        "ema": True,
        "ema_decay": 0.99,
        "swa": True,
        "default_dtype": "float32",
        "device": device,
        "enable_cueq": enable_cueq,
        "model_dir": str(paths.models),
        "checkpoints_dir": str(paths.models),
        "log_dir": str(paths.models),
        "results_dir": str(paths.models),
    }
    if spec.head is not None:
        config["foundation_head"] = spec.head
    return config


def train_single_head_finetune(
    paths: TutorialPaths,
    foundation_key: str,
    device: str,
    epochs: int,
    enable_cueq: bool = False,
    restart: bool = False,
) -> Path:
    """Fine-tune MP-0 or MH-1 on the molecular XTB target data."""
    spec = FOUNDATIONS[foundation_key]
    print_section("3", f"Fine-tune {spec.label} on XTB")
    require_files(
        paths.root,
        str(paths.train.relative_to(paths.root)),
        str(paths.valid.relative_to(paths.root)),
        str(paths.test.relative_to(paths.root)),
    )
    config_path = paths.configs / f"{spec.run_name}.yml"
    write_yaml(
        config_path,
        make_finetune_config(paths, foundation_key, device, epochs, enable_cueq),
    )
    print(f"Configuration -> {config_path}")
    print(f"Epochs: {epochs} | device: {device} | E0s: estimated")
    train(config_path, paths.root, restart=restart)
    model = find_model(paths.models, spec.run_name, spec.seed)
    print(f"Fine-tuned model -> {model}")
    return model


def prediction_metrics(reference: list, prediction: list) -> dict[str, float]:
    """Compute energy-per-atom and Cartesian-force RMSEs."""
    reference_energy = np.array(
        [atoms.info["energy_xtb"] / len(atoms) for atoms in reference]
    )
    predicted_energy = np.array(
        [atoms.info["MACE_energy"] / len(atoms) for atoms in prediction]
    )
    reference_forces = np.concatenate(
        [atoms.arrays["forces_xtb"] for atoms in reference]
    )
    predicted_forces = np.concatenate(
        [atoms.arrays["MACE_forces"] for atoms in prediction]
    )
    return {
        "energy_rmse_eV_per_atom": float(
            np.sqrt(np.mean((reference_energy - predicted_energy) ** 2))
        ),
        "force_rmse_eV_per_A": float(
            np.sqrt(np.mean((reference_forces - predicted_forces) ** 2))
        ),
    }


def compare_finetuned_models(
    paths: TutorialPaths, device: str, enable_cueq: bool = False
) -> None:
    """Evaluate available MP-0, MH-1, and replay models on one held-out test set."""
    print_section("4", "Compare fine-tuned foundation models")
    reference = read(paths.test, index=":")
    available: list[tuple[str, FoundationSpec, Path]] = []
    for key in ("mp0", "mh1", "replay"):
        spec = FOUNDATIONS[key]
        try:
            model = find_model(paths.models, spec.run_name, spec.seed)
        except FileNotFoundError:
            print(f"Skipping {spec.label}: no trained model found")
            continue
        available.append((key, spec, model))
    if not available:
        raise FileNotFoundError("Train at least one fine-tuned model before comparison")

    figure, axes = plt.subplots(len(available), 2, figsize=(10, 4 * len(available)))
    axes = np.atleast_2d(axes)
    metrics: dict[str, dict[str, float]] = {}
    reference_energy = np.array(
        [atoms.info["energy_xtb"] / len(atoms) for atoms in reference]
    )
    reference_forces = np.concatenate(
        [atoms.arrays["forces_xtb"] for atoms in reference]
    ).ravel()

    for row, (key, spec, model) in enumerate(available):
        output = paths.evaluations / f"{spec.run_name}_test.xyz"
        evaluate(paths.test, model, output, paths.root, device, enable_cueq)
        prediction = read(output, index=":")
        metrics[key] = prediction_metrics(reference, prediction)
        predicted_energy = np.array(
            [atoms.info["MACE_energy"] / len(atoms) for atoms in prediction]
        )
        predicted_forces = np.concatenate(
            [atoms.arrays["MACE_forces"] for atoms in prediction]
        ).ravel()

        for axis, x_value, y_value, quantity in (
            (axes[row, 0], reference_energy, predicted_energy, "Energy (eV/atom)"),
            (axes[row, 1], reference_forces, predicted_forces, "Force (eV/A)"),
        ):
            axis.scatter(x_value, y_value, s=7, alpha=0.35)
            low = min(float(x_value.min()), float(y_value.min()))
            high = max(float(x_value.max()), float(y_value.max()))
            axis.plot([low, high], [low, high], color="black", linewidth=1)
            axis.set(
                xlabel=f"XTB {quantity}",
                ylabel=f"MACE {quantity}",
                title=f"{spec.label}: {quantity}",
            )

    figure.tight_layout()
    comparison_plot = paths.plots / "finetuning_comparison.png"
    figure.savefig(comparison_plot, dpi=160)
    plt.close(figure)
    metrics_path = paths.evaluations / "finetuning_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print("\nmodel                              E RMSE (eV/atom)   F RMSE (eV/A)")
    for key, spec, _ in available:
        print(
            f"{spec.label:34s} "
            f"{metrics[key]['energy_rmse_eV_per_atom']:16.6f} "
            f"{metrics[key]['force_rmse_eV_per_A']:15.6f}"
        )
    print(f"Metrics -> {metrics_path}")
    print(f"Parity plots -> {comparison_plot}")


# %% [markdown]
# ## 5. Multihead replay fine-tuning
# Naive fine-tuning updates shared weights only on XTB and can forget broadly
# useful foundation behavior.  Replay adds a second ``pt_head``.  The target
# head learns real XTB labels, while selected MP configurations receive
# pseudolabels from the frozen starting model.  Shared representation weights
# see both losses, but separate readouts keep inconsistent energy references
# apart.  MACE automatically lowers the learning rate and enables a long-decay
# EMA unless ``force_mh_ft_lr`` is requested.


def make_replay_config(
    paths: TutorialPaths,
    device: str,
    epochs: int,
    replay_samples: int,
    enable_cueq: bool = False,
) -> dict[str, object]:
    """Construct the MACE 0.3.16 MH-1 pseudolabel-replay example."""
    spec = FOUNDATIONS["replay"]
    return {
        "model": "MACE",
        "foundation_model": spec.alias,
        "foundation_head": spec.head,
        "multiheads_finetuning": True,
        "pseudolabel_replay": True,
        "pt_train_file": "mp",
        "num_samples_pt": replay_samples,
        "subselect_pt": "random",
        "filter_type_pt": "combinations",
        "atomic_numbers": "[1, 6, 8]",
        "weight_pt_head": 1.0,
        "E0s": "estimated",
        "name": spec.run_name,
        "seed": spec.seed,
        "train_file": str(paths.train),
        "valid_file": str(paths.valid),
        "test_file": str(paths.test),
        "energy_key": "energy_xtb",
        "forces_key": "forces_xtb",
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 0.0,
        "batch_size": 5,
        "max_num_epochs": epochs,
        "swa": True,
        "default_dtype": "float32",
        "device": device,
        "enable_cueq": enable_cueq,
        "model_dir": str(paths.models),
        "checkpoints_dir": str(paths.models),
        "log_dir": str(paths.models),
        "results_dir": str(paths.models),
    }


def train_replay_finetune(
    paths: TutorialPaths,
    device: str,
    epochs: int,
    replay_samples: int,
    enable_cueq: bool = False,
    restart: bool = False,
) -> Path:
    """Fine-tune MH-1 with a small pedagogical pseudolabel replay subset."""
    print_section("5", "Fine-tune MH-1 with multihead replay")
    config_path = paths.configs / "ft_mh1_xtb_replay.yml"
    write_yaml(
        config_path,
        make_replay_config(
            paths, device, epochs, replay_samples, enable_cueq=enable_cueq
        ),
    )
    print(f"Configuration -> {config_path}")
    print(
        f"Replay samples: {replay_samples}. Production studies commonly use "
        "many thousands; this small value is for the lecture."
    )
    train(config_path, paths.root, restart=restart)
    spec = FOUNDATIONS["replay"]
    model = find_model(paths.models, spec.run_name, spec.seed)
    print(f"Multihead model -> {model}")
    print(
        "Use the Default target head for XTB deployment; pt_head preserves replay behavior."
    )
    return model


# %% [markdown]
# ## 6. Conditioning on total charge, spin, and other properties
# Geometry and elements do not uniquely define an electronic state.  MACE can
# embed graph-level quantities such as total charge, spin multiplicity, or
# electronic temperature.  These inputs must vary in the labeled training data;
# adding metadata to a neutral-only dataset cannot teach charged-state physics.


def write_conditioning_examples(paths: TutorialPaths) -> None:
    """Write inference metadata examples and a conditional-MACE YAML template."""
    print_section("6", "Condition MACE on charge, spin, and electronic temperature")
    require_files(paths.root, str(paths.valid.relative_to(paths.root)))
    geometry = read(paths.valid, index=0)
    examples = []
    for charge, spin, temperature in ((-1, 2, 300.0), (0, 1, 300.0), (1, 2, 600.0)):
        atoms = geometry.copy()
        atoms.calc = None
        atoms.info = {
            "total_charge": charge,
            "total_spin": spin,
            "elec_temp": float(temperature),
        }
        for key in list(atoms.arrays):
            if key not in {"numbers", "positions"}:
                del atoms.arrays[key]
        examples.append(atoms)
    example_path = paths.data / "conditioning_inputs.xyz"
    write(example_path, examples)

    conditional_config = {
        "name": "conditional_mace_example",
        "model": "MACE",
        "train_file": "REPLACE_WITH_CHARGE_SPIN_LABELED_DATA.xyz",
        "valid_fraction": 0.1,
        "energy_key": "energy",
        "forces_key": "forces",
        "total_charge_key": "total_charge",
        "total_spin_key": "total_spin",
        "elec_temp_key": "elec_temp",
        "embedding_specs": {
            "total_charge": {
                "type": "categorical",
                "per": "graph",
                "num_classes": 7,
                "offset": 3,
                "emb_dim": 32,
            },
            "total_spin": {
                "type": "categorical",
                "per": "graph",
                "num_classes": 5,
                "emb_dim": 32,
            },
            "elec_temp": {
                "type": "continuous",
                "per": "graph",
                "in_dim": 1,
                "emb_dim": 32,
            },
        },
        "use_embedding_readout": True,
        "E0s": "average",
        "r_max": 5.0,
        "num_channels": 64,
        "max_L": 1,
        "batch_size": 8,
        "max_num_epochs": 100,
    }
    config_path = paths.configs / "conditional_mace_template.yml"
    write_yaml(config_path, conditional_config)

    print(f"Three electronic-state inputs -> {example_path}")
    print(f"Conditional training template -> {config_path}")
    print(
        "Important: conditioning_inputs.xyz contains inputs only. Supply reference "
        "energies/forces computed for each charge, spin, and property value before training."
    )
    print(
        "For inference with a custom conditional model, pass info_keys to "
        "MACECalculator and set atoms.info['total_charge'], ['total_spin'], and ['elec_temp']."
    )


def run_omol_charge_example(paths: TutorialPaths, device: str) -> None:
    """Use the public charge/spin-aware MACE-OMOL foundation model."""
    print_section("7", "Evaluate charge and spin with MACE-OMOL")
    require_files(paths.root, str(paths.valid.relative_to(paths.root)))
    print(
        "This optional stage downloads the large ASL-licensed MACE-OMOL checkpoint. "
        "Continuing implies acceptance of its license terms."
    )
    calculator = mace_omol(model="extra_large", device=device, default_dtype="float32")
    geometry = read(paths.valid, index=0)
    results = []
    for charge, spin in ((-1.0, 2.0), (0.0, 1.0), (1.0, 2.0)):
        atoms = geometry.copy()
        # MACECalculator maps these convenient ASE info names to the model's
        # canonical total_charge and total_spin inputs.
        atoms.info["charge"] = charge
        atoms.info["spin"] = spin  # spin multiplicity: 1=singlet, 2=doublet
        atoms.calc = calculator
        energy = float(atoms.get_potential_energy())
        force_norm = float(np.linalg.norm(atoms.get_forces()))
        results.append(
            {
                "charge": charge,
                "spin_multiplicity": spin,
                "energy_eV": energy,
                "force_norm": force_norm,
            }
        )
        print(
            f"charge={charge:+.0f}, multiplicity={spin:.0f}: "
            f"E={energy:.6f} eV, ||F||={force_norm:.6f} eV/A"
        )
    output = paths.evaluations / "omol_charge_spin_example.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Results -> {output}")


# %% [markdown]
# # Stage dispatcher


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    root = args.work_dir.expanduser().resolve()
    set_project_caches(root)
    device = choose_device(args.device)
    paths = TutorialPaths.from_root(root)
    paths.create_output_dirs()

    print_section("", "MACE Foundation-Model Fine-Tuning")
    print(f"Stage: {args.stage} | device: {device} | work directory: {root}")

    if args.stage == "prepare":
        prepare_finetuning_data(paths)
    elif args.stage == "e0s":
        analyze_e0_reestimation(paths, device, args.enable_cueq)
    elif args.stage in ("mp0", "mh1"):
        train_single_head_finetune(
            paths,
            args.stage,
            device,
            args.epochs,
            args.enable_cueq,
            args.restart,
        )
    elif args.stage == "compare":
        compare_finetuned_models(paths, device, args.enable_cueq)
    elif args.stage == "replay":
        train_replay_finetune(
            paths,
            device,
            args.epochs,
            args.replay_samples,
            args.enable_cueq,
            args.restart,
        )
    elif args.stage == "properties":
        write_conditioning_examples(paths)
    elif args.stage == "omol":
        run_omol_charge_example(paths, device)


if __name__ == "__main__":
    main()
