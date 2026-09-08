#!/usr/bin/env python3
"""MACE in Practice I: fit, test, and run a molecular MACE model.

This is the batch/HPC companion to ``T01_MACE_Practice_I.ipynb``.  It follows
the same scientific story while replacing notebook magics and interactive
viewers with restartable stages and plots written to disk.

Learning objectives
-------------------
1. Inspect a molecular training database and prepare train/test subsets.
2. Understand the main MACE architecture and optimization parameters.
3. Train a MACE model and assess energy/force parity against XTB.
4. Use the trained model as an ASE calculator for molecular dynamics.

Run ``python T01_MACE_Practice_I.py --help`` for the stage interface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write
from hpc_common import (
    choose_device,
    configure_logging,
    energy_force_parity,
    evaluate,
    find_model,
    print_section,
    require_files,
    run_md,
    select_by_info,
    set_project_caches,
    train,
    write_yaml,
)

from mace.calculators import MACECalculator

MODEL_NAME = "mace01"
MODEL_SEED = 123

# --- Dataset sizes used in the lecture ---
# The cached XYZ starts with isolated H, C, and O atoms.  These three structures
# provide the atomic reference energies E0 used by MACE.
N_ISOLATED_ATOMS = 3
N_TRAINING_CONFIGS = 200
N_TEST_CONFIGS = 1000

# --- Small model used to keep the live tutorial affordable ---
N_CHANNELS = 32
CUTOFF_ANGSTROM = 4.0
BATCH_SIZE = 10


# %% [markdown]
# # Command-line interface
# The practical tutorial is split into independent HPC stages.  ``all`` is
# convenient for a local demonstration; separate jobs are safer on a cluster.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "train", "evaluate", "md"),
        default="all",
        help="Run the full workflow or one restartable stage.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.cwd(),
        help="Tutorial data/output root (must contain data/).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--md-steps", type=int, default=2000)
    parser.add_argument(
        "--restart", action="store_true", help="Restart latest MACE checkpoint."
    )
    parser.add_argument("--enable-cueq", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def paths(work_dir: Path) -> dict[str, Path]:
    """Keep every generated artifact below one user-selected work directory."""
    return {
        "config": work_dir / "config" / "config-02.yml",
        "model_dir": work_dir / "MACE_models",
        "train": work_dir / "data" / "solvent_xtb_train_200.xyz",
        "test": work_dir / "data" / "solvent_xtb_test.xyz",
        "plots": work_dir / "plots" / "T01",
    }


# %% [markdown]
# ## 1. Understand and prepare the data


def prepare(work_dir: Path, output_paths: dict[str, Path]) -> None:
    """Inspect the cached carbonate configurations and create data splits."""
    print_section("1", "Understand and prepare the data")
    require_files(
        work_dir,
        "data/solvent_configs.xyz",
        "data/solvent_molecs.xyz",
        "data/solvent_xtb.xyz",
    )
    output_paths["plots"].mkdir(parents=True, exist_ok=True)

    # Three versions of the same database are provided:
    #   * solvent_configs.xyz: geometries only
    #   * solvent_molecs.xyz: molecule counts/compositions added
    #   * solvent_xtb.xyz: XTB reference energies and forces added
    raw_configs = read(work_dir / "data" / "solvent_configs.xyz", index=":")
    molecule_configs = read(work_dir / "data" / "solvent_molecs.xyz", index=":")
    labeled_configs = read(work_dir / "data" / "solvent_xtb.xyz", index=":")
    print(f"Raw configurations: {len(raw_configs)}")
    print(
        f"Atoms/config: min={min(map(len, raw_configs))}, max={max(map(len, raw_configs))}"
    )

    # A model needs diverse cluster sizes, not merely many copies of one system.
    # Save the histogram because a batch job has no interactive display.
    molecule_counts = [atoms.info["Nmols"] for atoms in molecule_configs]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(molecule_counts, bins=np.arange(0.5, 7.5), rwidth=0.8)
    axis.set(xlabel="Molecules/config", ylabel="Configurations")
    figure.tight_layout()
    figure.savefig(output_paths["plots"] / "cluster_sizes.png", dpi=160)
    plt.close(figure)

    # Keep the three isolated atoms at the front of the training file so MACE
    # can read their E0 values.  The remaining 200 structures fit the potential.
    train_stop = N_ISOLATED_ATOMS + N_TRAINING_CONFIGS
    training_set = labeled_configs[:train_stop]
    test_set = labeled_configs[-N_TEST_CONFIGS:]
    write(output_paths["train"], training_set)
    write(output_paths["test"], test_set)
    print(f"Training set: {len(training_set)} structures -> {output_paths['train']}")
    print(f"Test set:     {len(test_set)} structures -> {output_paths['test']}")
    print(f"Cluster-size plot -> {output_paths['plots'] / 'cluster_sizes.png'}")


# %% [markdown]
# ## 2. Define and train the MACE model
# ``num_channels`` controls feature width, ``max_L`` controls the highest
# equivariant message order, and ``r_max`` defines the local environment.


def write_training_config(
    output_paths: dict[str, Path], device: str, epochs: int, enable_cueq: bool
) -> None:
    # Model architecture: deliberately small for a live training exercise.
    config = {
        "model": "MACE",
        "num_channels": N_CHANNELS,
        "max_L": 0,
        "r_max": CUTOFF_ANGSTROM,
        # Output locations and reproducibility.
        "name": MODEL_NAME,
        "model_dir": str(output_paths["model_dir"]),
        "log_dir": str(output_paths["model_dir"]),
        "checkpoints_dir": str(output_paths["model_dir"]),
        "results_dir": str(output_paths["model_dir"]),
        # Data and property names in the extended XYZ files.
        "train_file": str(output_paths["train"]),
        "valid_fraction": 0.10,
        "test_file": str(output_paths["test"]),
        "energy_key": "energy_xtb",
        "forces_key": "forces_xtb",
        # Optimization settings.
        "device": device,
        "default_dtype": "float32",
        "batch_size": BATCH_SIZE,
        "max_num_epochs": epochs,
        "swa": True,
        "seed": MODEL_SEED,
        "enable_cueq": enable_cueq,
    }
    write_yaml(output_paths["config"], config)


def train_model(
    work_dir: Path,
    output_paths: dict[str, Path],
    device: str,
    epochs: int,
    enable_cueq: bool,
    restart: bool,
) -> None:
    print_section("2", "Define and train the MACE model")
    require_files(work_dir, str(output_paths["train"].relative_to(work_dir)))
    write_training_config(output_paths, device, epochs, enable_cueq)
    print(f"Configuration -> {output_paths['config']}")
    print(f"Training for {epochs} epochs on {device}")
    train(output_paths["config"], work_dir, restart=restart)


# %% [markdown]
# ## 3. Evaluate energies and forces
# The diagonal in each parity plot is perfect agreement with XTB.  Comparing
# train and test plots reveals whether errors are dominated by under/overfitting.


def evaluate_model(
    work_dir: Path,
    output_paths: dict[str, Path],
    device: str,
    enable_cueq: bool,
) -> None:
    print_section("3", "Evaluate energies and forces")
    model = find_model(output_paths["model_dir"], MODEL_NAME, MODEL_SEED)
    print(f"Model -> {model}")
    output_dir = work_dir / "tests" / MODEL_NAME
    train_output = output_dir / "solvent_train.xyz"
    test_output = output_dir / "solvent_test.xyz"
    evaluate(
        output_paths["train"],
        model,
        train_output,
        work_dir,
        device,
        enable_cueq,
    )
    evaluate(
        output_paths["test"],
        model,
        test_output,
        work_dir,
        device,
        enable_cueq,
    )

    train_reference = read(output_paths["train"], index=":")
    train_prediction = read(train_output, index=":")
    test_reference = read(output_paths["test"], index=":")
    test_prediction = read(test_output, index=":")
    energy_force_parity(
        train_reference,
        train_prediction,
        output_paths["plots"] / "train_parity.png",
    )
    energy_force_parity(
        test_reference,
        test_prediction,
        output_paths["plots"] / "test_parity.png",
    )
    print(f"Train parity -> {output_paths['plots'] / 'train_parity.png'}")
    print(f"Test parity  -> {output_paths['plots'] / 'test_parity.png'}")


# %% [markdown]
# ## 4. Molecular dynamics with MACE
# First test stability on a hot isolated molecule, then test transfer from the
# cluster training data to a periodic liquid configuration.


def molecular_dynamics(
    work_dir: Path,
    output_paths: dict[str, Path],
    device: str,
    md_steps: int,
    enable_cueq: bool,
) -> None:
    print_section("4", "Molecular dynamics with MACE")
    require_files(work_dir, "data/solvent_molecs.xyz", "data/solvent_liquid.xyz")
    model = find_model(output_paths["model_dir"], MODEL_NAME, MODEL_SEED)
    calculator = MACECalculator(
        model_paths=str(model),
        device=device,
        default_dtype="float32",
        enable_cueq=enable_cueq,
    )

    # A deliberately high temperature is a useful stress test: unstable models
    # tend to visit unphysical configurations and fail quickly.
    molecules = read(work_dir / "data" / "solvent_molecs.xyz", index=":")
    single_molecules = select_by_info(molecules, "Nmols", 1)
    if not single_molecules:
        raise RuntimeError("No Nmols=1 structure found in solvent_molecs.xyz")
    run_md(
        single_molecules[0].copy(),
        calculator,
        work_dir / "moldyn" / "mace01_md.xyz",
        output_paths["plots"] / "molecule_md.png",
        temperature_k=1200,
        steps=md_steps,
    )

    # The liquid was not represented directly by the small cluster training set.
    # Successful dynamics therefore probe transfer to the condensed phase.
    liquid = read(work_dir / "data" / "solvent_liquid.xyz")
    liquid.center()
    run_md(
        liquid,
        calculator,
        work_dir / "moldyn" / "mace01_md_liquid.xyz",
        output_paths["plots"] / "liquid_md.png",
        temperature_k=500,
        steps=md_steps,
    )
    print(f"Molecule trajectory -> {work_dir / 'moldyn' / 'mace01_md.xyz'}")
    print(f"Liquid trajectory   -> {work_dir / 'moldyn' / 'mace01_md_liquid.xyz'}")


# %% [markdown]
# # Stage dispatcher
# The order below is the same order in which the notebook is taught.


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    work_dir = args.work_dir.expanduser().resolve()
    set_project_caches(work_dir)
    device = choose_device(args.device)
    output_paths = paths(work_dir)
    output_paths["model_dir"].mkdir(parents=True, exist_ok=True)
    output_paths["plots"].mkdir(parents=True, exist_ok=True)

    print_section("", "MACE in Practice I")
    print(f"Stage: {args.stage} | device: {device} | work directory: {work_dir}")

    if args.stage in ("all", "prepare"):
        prepare(work_dir, output_paths)
    if args.stage in ("all", "train"):
        train_model(
            work_dir,
            output_paths,
            device,
            args.epochs,
            args.enable_cueq,
            args.restart,
        )
    if args.stage in ("all", "evaluate"):
        evaluate_model(work_dir, output_paths, device, args.enable_cueq)
    if args.stage in ("all", "md"):
        molecular_dynamics(
            work_dir, output_paths, device, args.md_steps, args.enable_cueq
        )


if __name__ == "__main__":
    main()
