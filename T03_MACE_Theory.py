#!/usr/bin/env python3
"""Deep dive into the MACE architecture.

This script mirrors ``T03_MACE_Theory.ipynb`` while remaining runnable in a
headless batch job.  It builds a deliberately tiny, untrained MACE model and
walks through the tensors produced by its main architectural blocks.

Learning objectives
-------------------
1. See how spherical harmonics transform under rotation.
2. Construct rotation-invariant quantities with an e3nn tensor product.
3. Convert an atomic structure into MACE's neighbor graph representation.
4. Follow features through embedding, interaction, product, and readout blocks.
5. Visualize fixed radial bases and learned radial MLP functions.

The numerical values are not predictions: the model is intentionally untrained.
The tensor shapes and symmetry behavior are the lesson.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional
from ase.io import read

# MACE sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD before e3nn loads its constants.
# isort: off
from mace import data, modules, tools
from e3nn import o3

# isort: on

from hpc_common import (
    configure_logging,
    print_section,
    require_files,
    set_project_caches,
)
from scipy.spatial.transform import Rotation

# --- Tiny demonstration model ---
# Only H, C, and O occur in the carbonate molecule used by this tutorial.
ATOMIC_NUMBERS = [1, 6, 8]
ATOMIC_ENERGIES = np.array([-1.0, -3.0, -5.0])
CUTOFF_ANGSTROM = 3.0
N_CHANNELS = 8
MAX_ELL = 2
N_INTERACTIONS = 2


# %% [markdown]
# # Command-line interface
# Figures are written below ``plots/T03`` because compute nodes have no display.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# %% [markdown]
# ## 1. Define a small MACE model
# These are the same concepts exposed by ``mace_run_train``.  Small dimensions
# make every intermediate tensor easy to inspect during the lecture.


def build_model() -> tuple[modules.MACE, tools.AtomicNumberTable, float]:
    print_section("1", "Define a small, untrained MACE model")
    z_table = tools.AtomicNumberTable(ATOMIC_NUMBERS)
    config = {
        # Chemical species and per-element reference energies E0.
        "num_elements": len(ATOMIC_NUMBERS),
        "atomic_energies": ATOMIC_ENERGIES,
        "avg_num_neighbors": 8.0,
        "atomic_numbers": z_table.zs,
        # Radial and angular resolution of each local environment.
        "r_max": CUTOFF_ANGSTROM,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": MAX_ELL,
        # Two message-passing layers with residual equivariant interactions.
        "num_interactions": N_INTERACTIONS,
        "interaction_cls_first": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "interaction_cls": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        # Eight scalar and eight vector channels.  ``0e`` denotes even scalars;
        # ``1o`` denotes odd vectors.
        "hidden_irreps": o3.Irreps(f"{N_CHANNELS}x0e + {N_CHANNELS}x1o"),
        "correlation": 3,
        # The final nonlinear readout maps invariant features to atomic energy.
        "MLP_irreps": o3.Irreps("16x0e"),
        "gate": torch.nn.functional.silu,
    }
    model = modules.MACE(**config)
    print(f"Elements: {z_table.zs}")
    print(f"Cutoff: {CUTOFF_ANGSTROM} A")
    print(f"Hidden irreps: {config['hidden_irreps']}")
    print(f"Message-passing layers: {N_INTERACTIONS}")
    return model, z_table, CUTOFF_ANGSTROM


def spherical_harmonics_demo(output_dir: Path) -> None:
    """Demonstrate equivariance and an invariant tensor-product contraction."""
    print_section("2", "Spherical harmonics and rotational invariance")

    # For max_ell=2 the output contains 1 + 3 + 5 = 9 components: one scalar
    # (l=0), one vector-like triplet (l=1), and one rank-2 quintet (l=2).
    spherical_harmonics = o3.SphericalHarmonics(
        [0, 1, 2], normalize=True, normalization="component"
    )
    vector = torch.tensor([1.0, 0.2, 0.75])
    print("Spherical harmonics for [1.0, 0.2, 0.75]:", spherical_harmonics(vector))

    # Rotate one vector through 360 degrees.  Individual components change, but
    # they transform in a precisely prescribed equivariant way.
    axis = np.array([0.0, 0.7071, 0.7071])
    vectors = np.array(
        [
            Rotation.from_rotvec(angle * 2 * np.pi * axis / 360).as_matrix()
            @ vector.numpy()
            for angle in range(360)
        ]
    )
    values = spherical_harmonics(torch.from_numpy(vectors)).detach().numpy()
    labels = [
        f"l={ell}, m={order}" for ell in range(3) for order in range(-ell, ell + 1)
    ]
    figure, plot_axis = plt.subplots(figsize=(9, 5))
    plot_axis.plot(values, label=labels)
    plot_axis.set(xlabel="Rotation angle (degree)", ylabel="Spherical harmonic value")
    plot_axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    harmonics_plot = output_dir / "spherical_harmonics.png"
    figure.savefig(harmonics_plot, dpi=160)
    plt.close(figure)
    print(f"Rotation plot -> {harmonics_plot}")

    # A tensor product combines two equivariant representations.  Asking e3nn
    # for only 0e outputs contracts their angular information into invariants.
    rng = np.random.default_rng(0)
    vector_1 = rng.normal(size=3)
    vector_1 /= np.linalg.norm(vector_1)
    vector_2 = rng.normal(size=3)
    vector_2 /= np.linalg.norm(vector_2)
    harmonics_1 = spherical_harmonics(torch.from_numpy(vector_1))
    harmonics_2 = spherical_harmonics(torch.from_numpy(vector_2))

    tensor_product = o3.FullyConnectedTensorProduct(
        irreps_in1=o3.Irreps("1x0e + 1x1o + 1x2e"),
        irreps_in2=o3.Irreps("1x0e + 1x1o + 1x2e"),
        irreps_out=o3.Irreps("3x0e"),
        internal_weights=False,
    )
    weights = torch.arange(
        1, tensor_product.weight_numel + 1, dtype=torch.get_default_dtype()
    )
    invariant_before = tensor_product(
        harmonics_1.unsqueeze(0), harmonics_2.unsqueeze(0), weight=weights
    )
    # Apply exactly the same arbitrary rotation to both input vectors.  The
    # contracted scalar output should not change.
    rotation = Rotation.from_rotvec(77.7 * 2 * np.pi * axis / 360).as_matrix()
    invariant_after = tensor_product(
        spherical_harmonics(torch.from_numpy(rotation @ vector_1)).unsqueeze(0),
        spherical_harmonics(torch.from_numpy(rotation @ vector_2)).unsqueeze(0),
        weight=weights,
    )
    print("Invariant tensor product before rotation:", invariant_before)
    print("Invariant tensor product after rotation: ", invariant_after)
    if not torch.allclose(invariant_before, invariant_after, atol=1e-10):
        raise AssertionError("Tensor-product invariant changed under joint rotation")


# %% [markdown]
# ## 3. Convert atoms into a graph
# Atoms become nodes, while every pair within ``r_max`` becomes a directed edge.
# Edge shifts carry periodic-boundary information when a cell is present.


def atomic_graph(root: Path, z_table: tools.AtomicNumberTable, cutoff: float):
    print_section("3", "Represent an atomic structure as a neighbor graph")
    require_files(root, "data/solvent_rotated.xyz")
    molecule = read(root / "data" / "solvent_rotated.xyz", index=0)

    # Configuration is the lightweight, NumPy-level representation.  AtomicData
    # adds one-hot species and a cutoff neighbor list as PyTorch tensors.
    config = data.Configuration(
        atomic_numbers=molecule.numbers,
        positions=molecule.positions,
        properties={},
        property_weights={},
    )
    graph = data.AtomicData.from_config(config, z_table=z_table, cutoff=cutoff)
    print(f"Molecule: {molecule.get_chemical_formula()} ({len(molecule)} atoms)")
    print("positions:", graph.positions.shape)
    print("node_attrs:", graph.node_attrs.shape)
    print("edge_index:", graph.edge_index.shape)
    return graph


def plot_radial_basis(
    model: modules.MACE, graph, cutoff: float, output_dir: Path
) -> None:
    """Plot the fixed Bessel/cutoff features supplied to each interaction."""
    distances = torch.linspace(0.1, cutoff, 100).unsqueeze(-1)
    radial, _ = model.radial_embedding(
        distances, graph.node_attrs, graph.edge_index, model.atomic_numbers
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    for index in range(radial.shape[1]):
        axis.plot(
            distances.numpy(), radial[:, index].detach().numpy(), label=f"Basis {index}"
        )
    axis.set(xlabel="Distance (A)", ylabel="Value", title="MACE radial basis")
    axis.legend(fontsize=8)
    figure.tight_layout()
    output = output_dir / "radial_basis.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Fixed radial basis -> {output}")


def plot_learned_radials(
    model: modules.MACE,
    graph,
    cutoff: float,
    output_dir: Path,
    interaction_index: int,
) -> None:
    """Plot five outputs of one interaction block's untrained radial MLP."""
    distances = torch.linspace(0.1, cutoff, 100).unsqueeze(-1)
    edge_features, _ = model.radial_embedding(
        distances, graph.node_attrs, graph.edge_index, model.atomic_numbers
    )
    weights = (
        model.interactions[interaction_index]
        .conv_tp_weights(edge_features)
        .detach()
        .numpy()
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    for index in range(min(5, weights.shape[1])):
        axis.plot(distances.numpy(), weights[:, index], label=f"Weight {index}")
    axis.set(
        xlabel="Distance (A)",
        ylabel="Value",
        title=f"Untrained interaction {interaction_index + 1} radial MLP",
    )
    axis.legend()
    figure.tight_layout()
    output = output_dir / f"interaction_{interaction_index + 1}_radial_mlp.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Interaction {interaction_index + 1} radial MLP -> {output}")


# %% [markdown]
# ## 4. Follow one message-passing layer
# The sequence is embedding -> interaction -> product -> readout.  Printing the
# tensor shapes connects the equations in the notebook to the implementation.


def architecture_walkthrough(
    model: modules.MACE, graph, cutoff: float, output_dir: Path
) -> None:
    print_section("4", "Walk through MACE feature construction")

    # Edge geometry: R_ij vectors and their scalar lengths.
    vectors, lengths = modules.utils.get_edge_vectors_and_lengths(
        positions=graph.positions,
        edge_index=graph.edge_index,
        shifts=graph.shifts,
    )
    print(f"Graph has {graph.positions.shape[0]} nodes and {lengths.shape[0]} edges")
    # 1) Embedding: map one-hot chemical species to learned scalar channels.
    node_features = model.node_embedding(graph.node_attrs)

    # 2) Edge embedding: radial basis encodes distance; spherical harmonics
    # encode direction while preserving rotational equivariance.
    edge_features, cutoff_values = model.radial_embedding(
        lengths, graph.node_attrs, graph.edge_index, model.atomic_numbers
    )
    edge_attributes = model.spherical_harmonics(vectors)
    print("Initial node features:", node_features.shape)
    print("Radial edge features:", edge_features.shape)
    print("Spherical edge attributes:", edge_attributes.shape)

    # 3) Interaction: aggregate neighbor messages at each central atom.
    intermediate, skip_connection = model.interactions[0](
        node_feats=node_features,
        node_attrs=graph.node_attrs,
        edge_feats=edge_features,
        edge_attrs=edge_attributes,
        edge_index=graph.edge_index,
        cutoff=cutoff_values,
        first_layer=True,
    )
    print("Interaction output:", intermediate.shape)
    # 4) Product: build higher-body-order features on each atom.
    product_features = model.products[0](
        node_feats=intermediate,
        node_attrs=graph.node_attrs,
        sc=skip_connection,
    )
    print("Product features:", product_features.shape)
    # 5) Readout: convert invariant atomic features to per-atom energy.  This
    # demonstration model has one head, so every node receives head index zero.
    node_heads = torch.zeros(len(graph.positions), dtype=torch.long)
    layer_energy = model.readouts[0](product_features, node_heads)
    print("First readout per-atom energy:", layer_energy.shape)

    print_section("5", "Visualize radial features")
    plot_radial_basis(model, graph, cutoff, output_dir)
    for index in range(len(model.interactions)):
        plot_learned_radials(model, graph, cutoff, output_dir, index)


# %% [markdown]
# # Run the complete theory walkthrough


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    root = args.work_dir.expanduser().resolve()
    set_project_caches(root)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "plots" / "T03"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    print_section("", "Deep Dive into the MACE Architecture")
    print(f"Figures will be written to {output_dir}")

    model, z_table, cutoff = build_model()
    spherical_harmonics_demo(output_dir)
    graph = atomic_graph(root, z_table, cutoff)
    architecture_walkthrough(model, graph, cutoff, output_dir)
    print("\nTheory walkthrough complete.")


if __name__ == "__main__":
    main()
