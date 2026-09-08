#!/usr/bin/env python3
"""MACE in Practice II: iterative learning, committees, and fine-tuning.

This is the batch/HPC companion to ``T02_MACE_Practice_II.ipynb``.  Each stage
corresponds to a scientific step in the notebook and writes a checkpoint or XYZ
file consumed by the next step.

Learning objectives
-------------------
1. Diagnose an unstable model trained on very little data.
2. Label failed MD configurations with XTB and add informative examples.
3. Estimate uncertainty from a committee trained with different random seeds.
4. Compare training from scratch with a pre-trained MACE foundation model.
5. Fine-tune the foundation model to the XTB reference level.

Recommended stage order: ``prepare -> initial -> label -> iterative ->
committee -> foundation -> finetune``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write
from hpc_common import (
    choose_device,
    configure_logging,
    find_model,
    print_section,
    require_files,
    run_md,
    select_by_info,
    set_project_caches,
    train,
    write_yaml,
)
from tqdm import tqdm

from mace.calculators import MACECalculator, mace_mp


@dataclass(frozen=True)
class ModelSpec:
    """The identity and default training length of one committee member."""

    name: str
    seed: int
    epochs: int


# The three committee members see the same data but start from different random
# weights.  Their disagreement is used as a simple uncertainty estimate.
MODEL_SPECS = {
    "initial": ModelSpec("mace02_com1", seed=123, epochs=300),
    "generation_1": ModelSpec("mace02_com1_gen1", seed=123, epochs=500),
    "committee_2": ModelSpec("mace02_com2", seed=345, epochs=500),
    "committee_3": ModelSpec("mace02_com3", seed=567, epochs=500),
    "finetuned": ModelSpec("finetuned_MACE", seed=345, epochs=500),
}

# --- Iterative-learning choices used in the lecture ---
N_ISOLATED_ATOMS = 3
N_INITIAL_TRAIN = 20
N_VALIDATION = 30
FAILED_FRAME_START = 40
FAILED_FRAME_STRIDE = 5
N_FAILED_FRAMES = 3


# %% [markdown]
# # Command-line interface
# ``--epochs-scale 0.01`` is useful for checking the workflow without waiting
# for a pedagogically meaningful fit.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "prepare",
            "initial",
            "label",
            "iterative",
            "committee",
            "foundation",
            "finetune",
        ),
        default="all",
    )
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--md-steps", type=int, default=2000)
    parser.add_argument("--epochs-scale", type=float, default=1.0)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--enable-cueq", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def path_map(root: Path) -> dict[str, Path]:
    """Name the checkpoints that connect the restartable tutorial stages."""
    return {
        "models": root / "MACE_models",
        "plots": root / "plots" / "T02",
        "train20": root / "data" / "solvent_xtb_train_20.xyz",
        "train50": root / "data" / "solvent_xtb_train_50.xyz",
        "valid30": root / "data" / "solvent_xtb_valid_30.xyz",
        "test": root / "data" / "solvent_xtb_test.xyz",
        "labeled_md": root / "data" / "mace02_md_100_xtb.xyz",
        "train_gen1": root / "data" / "solvent_xtb_train_23_gen1.xyz",
    }


# %% [markdown]
# ## Shared training configuration
# All scratch models use the same architecture and data keys.  Only the random
# seed, training set, and number of epochs change between committee members.


def base_training_config(
    paths: dict[str, Path],
    model_name: str,
    seed: int,
    epochs: int,
    train_file: Path,
    device: str,
    enable_cueq: bool,
) -> dict[str, object]:
    return {
        # Small architecture: fast enough to train during the lecture.
        "model": "MACE",
        "num_channels": 32,
        "max_L": 0,
        "r_max": 4.0,
        # Run identity and output locations.
        "name": model_name,
        "model_dir": str(paths["models"]),
        "log_dir": str(paths["models"]),
        "checkpoints_dir": str(paths["models"]),
        "results_dir": str(paths["models"]),
        # The validation set is deliberately disjoint from the training set.
        "train_file": str(train_file),
        "valid_file": str(paths["valid30"]),
        "test_file": str(paths["test"]),
        "energy_key": "energy_xtb",
        "forces_key": "forces_xtb",
        # Optimization and reproducibility.
        "device": device,
        "default_dtype": "float32",
        "batch_size": 10,
        "max_num_epochs": epochs,
        "swa": True,
        "seed": seed,
        "enable_cueq": enable_cueq,
    }


# %% [markdown]
# ## 1. Prepare a deliberately small training problem
# We begin with only 20 molecular configurations so that the first model will
# encounter geometries outside its training distribution during hot MD.


def prepare(root: Path, paths: dict[str, Path]) -> None:
    """Create disjoint train/validation/test subsets from the cached XTB data."""
    print_section("1", "Prepare the iterative-learning datasets")
    require_files(root, "data/solvent_xtb.xyz", "data/solvent_molecs.xyz")
    configs = read(root / "data" / "solvent_xtb.xyz", index=":")

    # The first three entries are isolated H, C, and O reference atoms.  Count
    # molecular configurations separately when describing dataset sizes.
    train_stop = N_ISOLATED_ATOMS + N_INITIAL_TRAIN
    valid_stop = train_stop + N_VALIDATION
    write(paths["train50"], configs[:valid_stop])
    write(paths["train20"], configs[:train_stop])
    write(paths["valid30"], configs[train_stop:valid_stop])
    if not paths["test"].is_file():
        write(paths["test"], configs[-1000:])
    print(f"Initial training set -> {paths['train20']}")
    print(f"Disjoint validation -> {paths['valid30']}")
    print(f"Held-out test set    -> {paths['test']}")


def train_named_model(
    root: Path,
    paths: dict[str, Path],
    key: str,
    train_file: Path,
    device: str,
    epochs_scale: float,
    enable_cueq: bool,
    restart: bool,
) -> Path:
    spec = MODEL_SPECS[key]
    epochs = max(1, round(spec.epochs * epochs_scale))
    config_path = root / "config" / f"{spec.name}.yml"
    config = base_training_config(
        paths,
        spec.name,
        spec.seed,
        epochs,
        train_file,
        device,
        enable_cueq,
    )
    if key == "initial":
        config["batch_size"] = 5
    write_yaml(config_path, config)
    print(
        f"Training {spec.name} for {epochs} epochs "
        f"(seed={spec.seed}, device={device})"
    )
    print(f"Configuration -> {config_path}")
    train(config_path, root, restart=restart)
    return find_model(paths["models"], spec.name, spec.seed)


def first_molecule(root: Path):
    """Return one isolated carbonate molecule for comparable MD experiments."""
    configs = read(root / "data" / "solvent_molecs.xyz", index=":")
    singles = select_by_info(configs, "Nmols", 1)
    if not singles:
        raise RuntimeError("No Nmols=1 structure found in solvent_molecs.xyz")
    return singles[0].copy()


def model_calculator(
    model: Path | list[Path], device: str, enable_cueq: bool
) -> MACECalculator:
    """Build either a single-model or committee ASE calculator."""
    paths = [str(item) for item in model] if isinstance(model, list) else str(model)
    return MACECalculator(
        model_paths=paths,
        device=device,
        default_dtype="float32",
        enable_cueq=enable_cueq,
    )


# %% [markdown]
# ## 2. Train the initial model and stress-test it with hot MD
# This first model sees only 20 molecular configurations.  Running at 1200 K
# intentionally drives it toward poorly represented regions of configuration
# space, making model failure easy to diagnose.


def run_initial_md(
    root: Path,
    paths: dict[str, Path],
    device: str,
    steps: int,
    enable_cueq: bool,
) -> None:
    spec = MODEL_SPECS["initial"]
    model = find_model(paths["models"], spec.name, spec.seed)
    print(f"Model -> {model}")
    run_md(
        first_molecule(root),
        model_calculator(model, device, enable_cueq),
        root / "moldyn" / "mace02_md.xyz",
        paths["plots"] / "initial_md.png",
        temperature_k=1200,
        steps=steps,
    )
    print(f"Hot-MD trajectory -> {root / 'moldyn' / 'mace02_md.xyz'}")


# %% [markdown]
# ## 3. Label the failed trajectory with XTB
# XTB acts as the reference method.  Comparing MACE and XTB along the trajectory
# reveals where the model leaves its domain of validity.


def label_initial_md(root: Path, paths: dict[str, Path]) -> None:
    print_section("3", "Label the initial MD trajectory with XTB")
    try:
        from xtb.ase.calculator import XTB
    except ImportError as error:
        raise RuntimeError(
            "The label stage requires the optional xtb Python package installed by "
            "setup_hpc_env.sh."
        ) from error

    trajectory_path = root / "moldyn" / "mace02_md.xyz"
    require_files(root, str(trajectory_path.relative_to(root)))
    # Limiting the expensive reference calculation to 100 saved frames keeps
    # this stage short enough for a tutorial job.
    trajectory = read(trajectory_path, index=":")[:100]
    calculator = XTB(method="GFN2-xTB")
    for atoms in tqdm(trajectory, desc="XTB labels"):
        atoms.calc = calculator
        atoms.info["energy_xtb"] = atoms.get_potential_energy()
        atoms.arrays["forces_xtb"] = atoms.get_forces()
        atoms.calc = None
    write(paths["labeled_md"], trajectory)

    figure, axis = plt.subplots(figsize=(7, 4))
    x_axis = np.arange(len(trajectory))
    axis.plot(
        x_axis, [at.info["energy_xtb"] / len(at) for at in trajectory], label="XTB"
    )
    axis.plot(
        x_axis, [at.info["energy_mace"] / len(at) for at in trajectory], label="MACE"
    )
    axis.set(xlabel="Saved MD frame", ylabel="Energy (eV/atom)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["plots"] / "initial_md_xtb_comparison.png", dpi=160)
    plt.close(figure)
    print(f"XTB-labeled trajectory -> {paths['labeled_md']}")
    print("Energy comparison -> " f"{paths['plots'] / 'initial_md_xtb_comparison.png'}")


def build_generation_one(paths: dict[str, Path]) -> None:
    """Add three representative failed MD frames to the initial training set."""
    base = read(paths["train20"], index=":")
    trajectory = read(paths["labeled_md"], index=":")

    # Select frames 40, 45, and 50.  Nearby frames are strongly correlated, so
    # spacing them is more informative than taking three consecutive snapshots.
    selection_stop = FAILED_FRAME_START + FAILED_FRAME_STRIDE * N_FAILED_FRAMES
    selected_indices = list(
        range(FAILED_FRAME_START, selection_stop, FAILED_FRAME_STRIDE)
    )
    selected = trajectory[FAILED_FRAME_START:selection_stop:FAILED_FRAME_STRIDE]
    write(paths["train_gen1"], base + selected)
    print(f"Added frames {selected_indices} -> {paths['train_gen1']}")


# %% [markdown]
# ## 4. Iterative training
# Retrain after adding only three high-value failure configurations, then repeat
# the same hot-MD stress test to see whether stability improves.


def run_iterative_stage(
    root: Path,
    paths: dict[str, Path],
    device: str,
    steps: int,
    epochs_scale: float,
    enable_cueq: bool,
    restart: bool,
) -> None:
    print_section("4", "Retrain with informative failed configurations")
    require_files(root, str(paths["labeled_md"].relative_to(root)))
    build_generation_one(paths)
    model = train_named_model(
        root,
        paths,
        "generation_1",
        paths["train_gen1"],
        device,
        epochs_scale,
        enable_cueq,
        restart,
    )
    run_md(
        first_molecule(root),
        model_calculator(model, device, enable_cueq),
        root / "moldyn" / "mace02_md_gen1.xyz",
        paths["plots"] / "generation_one_md.png",
        temperature_k=1200,
        steps=steps,
    )
    print(f"Generation-one trajectory -> {root / 'moldyn' / 'mace02_md_gen1.xyz'}")


def committee_models(paths: dict[str, Path]) -> list[Path]:
    """Return the three independently initialized generation-one models."""
    models = []
    for key in ("committee_3", "committee_2", "generation_1"):
        spec = MODEL_SPECS[key]
        models.append(find_model(paths["models"], spec.name, spec.seed))
    return models


def analyze_committee(paths: dict[str, Path], device: str, enable_cueq: bool) -> None:
    """Compare the committee mean/disagreement with the XTB reference."""
    trajectory = read(paths["labeled_md"], index=":")
    calculator = model_calculator(committee_models(paths), device, enable_cueq)
    committee_energies = []
    variances = []
    references = []
    for atoms in tqdm(trajectory, desc="MACE committee"):
        atoms.calc = calculator
        atoms.get_potential_energy()

        # In MACE 0.3.16 ``energy_comm`` contains one total energy per
        # committee member.  ``get_potential_energies()`` instead means
        # per-atom node energies and must not be used for this purpose.
        energies = np.asarray(calculator.results["energy_comm"], dtype=float)
        committee_energies.append(energies / len(atoms))

        # Variance of E/N is variance(E) / N^2.
        variances.append(float(calculator.results["energy_var"]) / len(atoms) ** 2)
        references.append(atoms.info["energy_xtb"] / len(atoms))
        atoms.calc = None

    committee_array = np.asarray(committee_energies)
    x_axis = np.arange(len(trajectory))
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(x_axis, references, label="XTB", color="black")
    for index in range(committee_array.shape[1]):
        axes[0].plot(x_axis, committee_array[:, index], label=f"MACE {index + 1}")
    axes[0].set_ylabel("Energy (eV/atom)")
    axes[0].legend()
    axes[1].plot(x_axis, variances)
    axes[1].set(xlabel="Saved MD frame", ylabel="Committee variance (eV/atom)^2")
    figure.tight_layout()
    figure.savefig(paths["plots"] / "committee.png", dpi=160)
    plt.close(figure)
    print(f"Committee comparison -> {paths['plots'] / 'committee.png'}")


# %% [markdown]
# ## 5. Committee uncertainty
# Train two more models on exactly the same data with different seeds.  Agreement
# is not a rigorous error bar, but large disagreement is a useful signal that a
# configuration lies outside the shared training distribution.


def run_committee_stage(
    root: Path,
    paths: dict[str, Path],
    device: str,
    epochs_scale: float,
    enable_cueq: bool,
    restart: bool,
) -> None:
    print_section("5", "Estimate uncertainty with a model committee")
    require_files(root, str(paths["train_gen1"].relative_to(root)))
    for key in ("committee_2", "committee_3"):
        train_named_model(
            root,
            paths,
            key,
            paths["train_gen1"],
            device,
            epochs_scale,
            enable_cueq,
            restart,
        )
    analyze_committee(paths, device, enable_cueq)


# %% [markdown]
# ## 6. Use a pre-trained MACE foundation model
# ``mace_mp(model="small")`` downloads a model trained on a much broader dataset.
# Here it is used without additional fitting to demonstrate transfer learning.


def run_foundation_stage(
    root: Path,
    paths: dict[str, Path],
    device: str,
    steps: int,
    enable_cueq: bool,
) -> None:
    print_section("6", "Run MD with the MACE-MP foundation model")
    print("Loading the small MACE-MP foundation model (downloaded on first use)")
    calculator = mace_mp(
        model="small",
        device=device,
        default_dtype="float32",
        enable_cueq=enable_cueq,
    )
    run_md(
        first_molecule(root),
        calculator,
        root / "moldyn" / "mace_mp_small_md.xyz",
        paths["plots"] / "mace_mp_small_md.png",
        temperature_k=1200,
        steps=steps,
    )
    print(f"Foundation-model trajectory -> {root / 'moldyn' / 'mace_mp_small_md.xyz'}")


# %% [markdown]
# ## 7. Fine-tune the foundation model to XTB
# Fine-tuning starts from learned representations rather than random weights.
# The force weight is increased because accurate gradients are essential for MD.


def run_finetune_stage(
    root: Path,
    paths: dict[str, Path],
    device: str,
    steps: int,
    epochs_scale: float,
    enable_cueq: bool,
    restart: bool,
) -> None:
    print_section("7", "Fine-tune MACE-MP to the XTB reference level")
    spec = MODEL_SPECS["finetuned"]
    config = {
        # Initialize from the pre-trained small MACE-MP model.
        "model": "MACE",
        "foundation_model": "small",
        "multiheads_finetuning": False,
        # Loss weights for molecular dynamics quality.
        "stress_weight": 0.0,
        "forces_weight": 10.0,
        "energy_weight": 1.0,
        # Run identity and output locations.
        "name": spec.name,
        "model_dir": str(paths["models"]),
        "log_dir": str(paths["models"]),
        "checkpoints_dir": str(paths["models"]),
        "results_dir": str(paths["models"]),
        # Target XTB dataset.
        "train_file": str(paths["train50"]),
        "valid_fraction": 0.10,
        "test_file": str(paths["test"]),
        "energy_key": "energy_xtb",
        "forces_key": "forces_xtb",
        # Optimization and reproducibility.
        "device": device,
        "default_dtype": "float32",
        "batch_size": 10,
        "max_num_epochs": max(1, round(spec.epochs * epochs_scale)),
        "swa": True,
        "seed": spec.seed,
        "enable_cueq": enable_cueq,
    }
    config_path = root / "config" / "config-07.yml"
    write_yaml(config_path, config)
    print(f"Fine-tuning configuration -> {config_path}")
    train(config_path, root, restart=restart)
    model = find_model(paths["models"], spec.name, spec.seed)
    run_md(
        first_molecule(root),
        model_calculator(model, device, enable_cueq),
        root / "moldyn" / "mace_finetuned_md.xyz",
        paths["plots"] / "mace_finetuned_md.png",
        temperature_k=1200,
        steps=steps,
    )
    print(f"Fine-tuned trajectory -> {root / 'moldyn' / 'mace_finetuned_md.xyz'}")


# %% [markdown]
# # Stage dispatcher
# Reading this block from top to bottom gives the complete notebook workflow.


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    root = args.work_dir.expanduser().resolve()
    set_project_caches(root)
    device = choose_device(args.device)
    paths = path_map(root)
    paths["models"].mkdir(parents=True, exist_ok=True)
    paths["plots"].mkdir(parents=True, exist_ok=True)

    print_section("", "MACE in Practice II")
    print(f"Stage: {args.stage} | device: {device} | work directory: {root}")

    if args.stage in ("all", "prepare"):
        prepare(root, paths)
    if args.stage in ("all", "initial"):
        print_section("2", "Stress-test the initial 20-configuration model")
        train_named_model(
            root,
            paths,
            "initial",
            paths["train20"],
            device,
            args.epochs_scale,
            args.enable_cueq,
            args.restart,
        )
        run_initial_md(root, paths, device, args.md_steps, args.enable_cueq)
    if args.stage in ("all", "label"):
        label_initial_md(root, paths)
    if args.stage in ("all", "iterative"):
        run_iterative_stage(
            root,
            paths,
            device,
            args.md_steps,
            args.epochs_scale,
            args.enable_cueq,
            args.restart,
        )
    if args.stage in ("all", "committee"):
        run_committee_stage(
            root,
            paths,
            device,
            args.epochs_scale,
            args.enable_cueq,
            args.restart,
        )
    if args.stage in ("all", "foundation"):
        run_foundation_stage(root, paths, device, args.md_steps, args.enable_cueq)
    if args.stage in ("all", "finetune"):
        run_finetune_stage(
            root,
            paths,
            device,
            args.md_steps,
            args.epochs_scale,
            args.enable_cueq,
            args.restart,
        )


if __name__ == "__main__":
    main()
