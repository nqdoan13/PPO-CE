# Git Guide

This folder contains two groups of scripts:

- dataset converters that turn raw graph data into ESG JSON or JSONL
- PPO-CE training code that learns to select ESG subgraphs

## Folder Layout

- `loadJSON.py`: convert GQA scene graphs to ESG
- `convert_3dssg_to_esg.py`: convert 3DSSG objects and relationships to ESG
- `convert_mutag_to_esg.py`: convert MUTAG to ESG
- `convert_yeasth_to_esg.py`: convert YeastH TU-format data to ESG
- `convert_md17_to_esg.py`: convert MD-17 style datasets such as `aspirin` and `benzene` to ESG
- `PPO/train_ppo.py`: train PPO-CE on ESG files
- `PPO/eval_ppo.py`: evaluate a trained PPO-CE checkpoint
- `PPO/model.py`: policy/value network used by PPO-CE
- `PPO/esg_env.py`: ESG environment and reward logic used by PPO-CE
- `PPO/requirements-rl.txt`: minimal RL dependencies

## Environment

Recommended Python packages:

```powershell
pip install -r Git/PPO/requirements-rl.txt
```

For `convert_mutag_to_esg.py`, you also need `torch_geometric`.

## ESG Output Format

All converters write ESG records with the same core shape:

```json
{
  "esg_id": "example",
  "entities": [],
  "relations": []
}
```

Some datasets also include `graph_label`.

## Dataset Conversion

### GQA

Use `loadJSON.py` for GQA scene graphs.

Input:

- a GQA scene-graph JSON file such as `val_sceneGraphs.json`

Example:

```powershell
python Git/loadJSON.py --file val_sceneGraphs.json --out val_esg_output.json
```

Useful options:

- `--count`: print the first few converted ESGs
- `--min-edge-confidence`: drop weak relation edges

### 3DSSG

Use `convert_3dssg_to_esg.py`.

Input:

- `objects.json`
- `relationships.json`

Example:

```powershell
python Git/convert_3dssg_to_esg.py `
  --objects Data/3DSSG/objects.json `
  --relationships Data/3DSSG/relationships.json `
  --out esg_3dssg_output.json
```

Useful options:

- `--count`: print the first N ESGs
- `--limit`: cap the number of scans
- `--min-edge-confidence`: filter relations

### MUTAG

Use `convert_mutag_to_esg.py`.

This script downloads or loads MUTAG through PyTorch Geometric.

Example:

```powershell
python Git/convert_mutag_to_esg.py --output-file Data/MUTAG/mutag_esg.json
```

Useful options:

- `--root`: PyG dataset cache root
- `--limit-graphs`: convert only part of the dataset

### YeastH

Use `convert_yeasth_to_esg.py`.

Expected input files are under `Data/YeastH/`.

Example:

```powershell
python Git/convert_yeasth_to_esg.py --format jsonl
```

Useful options:

- `--output-file`: override output path
- `--limit-graphs`: convert only part of the dataset
- `--format jsonl|json`: choose output format

### MD-17

Use `convert_md17_to_esg.py`.

Supported datasets:

- `aspirin`
- `benzene`

Example:

```powershell
python Git/convert_md17_to_esg.py --dataset aspirin --format jsonl
```

Useful options:

- `--output-file`: override output path
- `--limit-graphs`: convert only part of the dataset
- `--format jsonl|json`: choose output format

## PPO-CE Training

`PPO/train_ppo.py` trains on ESG files in `.json` or `.jsonl` format.

Recommended command:

```powershell
python Git/PPO/train_ppo.py `
  --esg-file train_esg_output.json `
  --checkpoint-dir checkpoints_train_esg `
  --metrics-out train_metrics.json `
  --rewards-out train_rewards.json
```

Important options:

- `--esg-file`: input ESG dataset
- `--limit-graphs`: cap dataset size
- `--max-nodes`: maximum nodes exposed to the policy
- `--max-steps`: maximum actions before forced stop
- `--updates`: number of PPO updates
- `--target-episodes`: optional early stopping by completed episodes
- `--device cuda|cpu`: preferred device

Output:

- checkpoint file: `ppo_esg.pt`
- optional metrics JSON
- optional rewards JSON

## PPO-CE Evaluation

`PPO/eval_ppo.py` evaluates a saved checkpoint on ESG files.

Example:

```powershell
python Git/PPO/eval_ppo.py `
  --esg-file val_esg_output.json `
  --checkpoint checkpoints_train_esg/ppo_esg.pt `
  --policy-mode greedy `
  --csv-out eval_rows.csv `
  --metrics-out eval_metrics.json
```

Important options:

- `--checkpoint`: path to trained PPO-CE checkpoint
- `--policy-mode greedy|sample`: action selection mode
- `--max-episodes`: optional episode cap
- `--limit-graphs`: optional input cap
- `--max-nodes` and `--max-steps`: optional runtime overrides

## Working Notes

- Run converters from the workspace root with explicit paths when possible.
- `PPO/train_ppo.py` and `PPO/eval_ppo.py` import files from the same `PPO` folder, so run them as scripts, for example `python Git/PPO/train_ppo.py`.
- If a dataset is large, prefer `jsonl` where the converter supports it.
- Use explicit output paths to avoid confusion about where files are written.

## Quick Start

Convert GQA:

```powershell
python Git/loadJSON.py --file val_sceneGraphs.json --out val_esg_output.json
```

Train PPO-CE:

```powershell
python Git/PPO/train_ppo.py --esg-file val_esg_output.json --checkpoint-dir checkpoints_train_esg
```

Evaluate PPO-CE:

```powershell
python Git/PPO/eval_ppo.py --esg-file val_esg_output.json --checkpoint checkpoints_train_esg/ppo_esg.pt
```
