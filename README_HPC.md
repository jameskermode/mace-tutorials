# MACE tutorials on HPC

Start with the student-facing walkthrough:

- [`STUDENT_GUIDE.md`](STUDENT_GUIDE.md): searchable commands, learning questions, and troubleshooting
- [`STUDENT_GUIDE.pdf`](STUDENT_GUIDE.pdf): printable A4 handout

These scripts are headless companions to the four notebooks:

- `T01_MACE_Practice_I.py`: data preparation, training, evaluation, and MD
- `T02_MACE_Practice_II.py`: iterative training, committee uncertainty, and fine-tuning
- `T03_MACE_Theory.py`: architecture walkthrough and saved figures
- `T04_MACE_Finetuning.py`: E0 re-estimation, MP-0/MH-1 comparison, replay, and property conditioning

They target `mace-torch==0.3.16` with PyTorch 2.6-2.8. Interactive molecule viewers were removed because
they require a browser; trajectories remain available as `extxyz`, and all plots are
saved below the selected work directory.

## Install

Run this on an internet-enabled login node. The environment is created on project
storage, not in `$HOME`. If `uv` is unavailable, the setup installs its standalone
binary below `/project/home/p201433/.local/bin`:

```bash
bash setup_hpc_env.sh
source /project/home/p201433/.venv-mace-tutorials/bin/activate
```

By default, the setup also clones the cached tutorial data to
`/project/home/p201433/Tutorials`. Set `DOWNLOAD_TUTORIAL_DATA=0` when the data is
already present, or set `DATA_DIR` to its location.

## Run

The practical tutorials expose restartable stages, which is preferable to one long
batch job:

```bash
python T01_MACE_Practice_I.py --work-dir /project/home/p201433/Tutorials --stage prepare
python T01_MACE_Practice_I.py --work-dir /project/home/p201433/Tutorials --stage train --device cuda
python T01_MACE_Practice_I.py --work-dir /project/home/p201433/Tutorials --stage evaluate --device cuda
python T01_MACE_Practice_I.py --work-dir /project/home/p201433/Tutorials --stage md --device cuda

python T02_MACE_Practice_II.py --help
python T03_MACE_Theory.py --work-dir /project/home/p201433/Tutorials
python T04_MACE_Finetuning.py --help
```

For Slurm, adjust the partition/resource directives in `run_tutorial.slurm`, then:

```bash
TUTORIAL=T01_MACE_Practice_I.py STAGE=train sbatch run_tutorial.slurm

# One-epoch T04 smoke tests before committing to full fine-tuning
TUTORIAL=T04_MACE_Finetuning.py STAGE=mp0 \
  TUTORIAL_EXTRA_ARGS="--epochs 1" sbatch run_tutorial.slurm
TUTORIAL=T04_MACE_Finetuning.py STAGE=mh1 \
  TUTORIAL_EXTRA_ARGS="--epochs 1" sbatch run_tutorial.slurm

# Small replay smoke test; increase both values for a meaningful run
TUTORIAL=T04_MACE_Finetuning.py STAGE=replay \
  TUTORIAL_EXTRA_ARGS="--epochs 1 --replay-samples 3" sbatch run_tutorial.slurm
```

The included Slurm resources follow the project SO3LR tutorial convention but use
one GPU and one task because these MACE tutorial stages are single-GPU processes.

Pass `--restart` to a training stage to use its latest MACE checkpoint. Pass
`--enable-cueq` only after installing the CuEq wheel matching the cluster CUDA
version, as shown in `setup_hpc_env.sh`.
