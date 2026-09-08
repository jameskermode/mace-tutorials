"""Shared helpers for running the MACE tutorials without Jupyter."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from ase import Atoms, units
from ase.constraints import FixCom
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

LOGGER = logging.getLogger("mace_tutorial")


def print_section(number: str, title: str) -> None:
    """Print a notebook-like section heading in batch logs."""
    heading = f"{number}. {title}" if number else title
    rule = "=" * len(heading)
    print(f"\n{heading}\n{rule}", flush=True)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def choose_device(requested: str) -> str:
    """Resolve ``auto`` while rejecting unavailable accelerators early."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but PyTorch cannot see a GPU")
    return requested


def require_files(root: Path, *relative_paths: str) -> None:
    missing = [path for path in relative_paths if not (root / path).is_file()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"Missing tutorial data below {root}:\n  - {formatted}\n"
            "Run setup_hpc_env.sh with DOWNLOAD_TUTORIAL_DATA=1, or pass "
            "--work-dir pointing at the imagdau/Tutorials checkout."
        )


def write_yaml(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(values, handle, sort_keys=False)
    LOGGER.info("Wrote %s", path)


def run_module(module: str, *arguments: str, cwd: Path) -> None:
    command = [sys.executable, "-m", module, *map(str, arguments)]
    LOGGER.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def train(config_path: Path, work_dir: Path, restart: bool = False) -> None:
    arguments = ["--config", str(config_path)]
    if restart:
        arguments.append("--restart_latest")
    run_module("mace.cli.run_train", *arguments, cwd=work_dir)


def evaluate(
    configs: Path,
    model: Path,
    output: Path,
    work_dir: Path,
    device: str,
    enable_cueq: bool = False,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    arguments = [
        "--configs",
        str(configs),
        "--model",
        str(model),
        "--output",
        str(output),
        "--device",
        device,
        "--default_dtype",
        "float32",
    ]
    if enable_cueq:
        arguments.append("--enable_cueq")
    run_module("mace.cli.eval_configs", *arguments, cwd=work_dir)


def find_model(model_dir: Path, name: str, seed: int) -> Path:
    """Prefer the SWA model and tolerate output naming changes across MACE patches."""
    candidates = [
        model_dir / f"{name}_run-{seed}_stagetwo.model",
        model_dir / f"{name}_run-{seed}.model",
    ]
    candidates.extend(sorted(model_dir.glob(f"{name}*{seed}*.model")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No trained model for name={name!r}, seed={seed} in {model_dir}"
    )


def run_md(
    atoms: Atoms,
    calculator: Any,
    trajectory: Path,
    plot_path: Path,
    temperature_k: float,
    steps: int,
    interval: int = 10,
    seed: int = 701,
) -> None:
    """Run deterministic Langevin MD and save trajectory plus diagnostics."""
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory.unlink(missing_ok=True)

    atoms.calc = calculator
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=rng)
    Stationary(atoms)
    if not np.any(atoms.pbc):
        ZeroRotation(atoms)
    if not any(isinstance(constraint, FixCom) for constraint in atoms.constraints):
        atoms.set_constraint([*atoms.constraints, FixCom()])
    dynamics = Langevin(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=temperature_k,
        friction=0.1 / units.fs,
        fixcm=False,
        rng=rng,
    )

    times: list[float] = []
    temperatures: list[float] = []
    energies: list[float] = []

    def save_frame() -> None:
        potential_energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        atoms.info["energy_mace"] = potential_energy
        atoms.arrays["forces_mace"] = forces
        atoms.write(trajectory, append=True, write_results=False)
        times.append(dynamics.get_time() / units.fs)
        temperatures.append(atoms.get_temperature())
        energies.append(potential_energy / len(atoms))

    dynamics.attach(save_frame, interval=interval)
    started = time.monotonic()
    dynamics.run(steps)
    LOGGER.info("MD finished in %.2f minutes", (time.monotonic() - started) / 60)

    figure, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(times, energies)
    axes[0].set_ylabel("E (eV/atom)")
    axes[1].plot(times, temperatures, color="tab:red")
    axes[1].set_ylabel("T (K)")
    axes[1].set_xlabel("Time (fs)")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def select_by_info(configs: list[Atoms], key: str, value: Any) -> list[Atoms]:
    return [atoms for atoms in configs if atoms.info.get(key) == value]


def energy_force_parity(
    reference: list[Atoms],
    predicted: list[Atoms],
    output: Path,
    reference_energy_key: str = "energy_xtb",
    reference_forces_key: str = "forces_xtb",
) -> None:
    """Save simple parity plots without relying on notebook-only aseMolec helpers."""
    if len(reference) != len(predicted):
        raise ValueError("Reference and prediction trajectories have different lengths")
    ref_energy = np.array(
        [atoms.info[reference_energy_key] / len(atoms) for atoms in reference]
    )
    pred_energy = np.array(
        [
            atoms.info.get("MACE_energy", atoms.info.get("energy_mace")) / len(atoms)
            for atoms in predicted
        ]
    )
    ref_forces = np.concatenate(
        [atoms.arrays[reference_forces_key] for atoms in reference]
    )
    pred_forces = np.concatenate(
        [
            atoms.arrays.get("MACE_forces", atoms.arrays.get("forces_mace"))
            for atoms in predicted
        ]
    )

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, x_value, y_value, label in (
        (axes[0], ref_energy, pred_energy, "Energy (eV/atom)"),
        (axes[1], ref_forces.ravel(), pred_forces.ravel(), "Force (eV/A)"),
    ):
        axis.scatter(x_value, y_value, s=8, alpha=0.5)
        low = min(float(np.min(x_value)), float(np.min(y_value)))
        high = max(float(np.max(x_value)), float(np.max(y_value)))
        axis.plot([low, high], [low, high], color="black", linewidth=1)
        rmse = float(np.sqrt(np.mean((x_value - y_value) ** 2)))
        axis.set(
            xlabel=f"XTB {label}", ylabel=f"MACE {label}", title=f"RMSE {rmse:.3g}"
        )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def set_project_caches(project_dir: Path) -> None:
    """Keep model and plotting caches on project storage, not a small home quota."""
    cache_dir = project_dir / ".cache"
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
