import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.distributions import Categorical

from esg_env import ESGSubgraphEnv
from model import ESGPolicyValueNet


def set_seed(seed: int):
    """Set Python, NumPy, and PyTorch seeds.

    Args:
        seed: Random seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_esgs(path: str) -> List[Dict]:
    """Load ESG records from JSON or JSONL.

    Args:
        path: Input ESG file path.
    """

    if path.lower().endswith(".jsonl"):
        esgs = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    esgs.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as handle:
            esgs = json.load(handle)
        if not isinstance(esgs, list):
            raise ValueError("Expected ESG JSON file as a list.")
    if not esgs:
        raise ValueError("No ESG entries found.")
    return esgs


def obs_to_tensor(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert one environment observation to batched tensors.

    Args:
        obs: Observation dictionary from the environment.
        device: Target PyTorch device.
    """

    return {
        "node_feats": torch.tensor(obs["node_feats"], dtype=torch.float32, device=device).unsqueeze(0),
        "label_ids": torch.tensor(obs["label_ids"], dtype=torch.long, device=device).unsqueeze(0),
        "selected_mask": torch.tensor(obs["selected_mask"], dtype=torch.float32, device=device).unsqueeze(0),
        "valid_mask": torch.tensor(obs["valid_mask"], dtype=torch.float32, device=device).unsqueeze(0),
        "action_mask": torch.tensor(obs["action_mask"], dtype=torch.float32, device=device).unsqueeze(0),
    }


def resolve_runtime_config(args, checkpoint: Dict) -> Dict:
    """Resolve evaluation settings from the checkpoint and CLI arguments.

    Args:
        args: Parsed command-line arguments.
        checkpoint: Loaded checkpoint dictionary.
    """

    config = dict(checkpoint.get("config", {}))
    return {
        "max_nodes": int(args.max_nodes if args.max_nodes > 0 else config.get("max_nodes", 32)),
        "max_steps": int(args.max_steps if args.max_steps > 0 else config.get("max_steps", 8)),
        "label_emb_dim": int(config.get("label_emb_dim", 32)),
        "hidden_dim": int(config.get("hidden_dim", 128)),
        "terminal_cov_weight": float(config.get("terminal_cov_weight", 1.5)),
        "terminal_attr_weight": float(config.get("terminal_attr_weight", 0.5)),
        "terminal_size_penalty": float(config.get("terminal_size_penalty", 0.5)),
        "step_cov_weight": float(config.get("step_cov_weight", 1.0)),
        "step_attr_weight": float(config.get("step_attr_weight", 0.5)),
        "step_size_penalty": float(config.get("step_size_penalty", 0.2)),
        "no_gain_penalty": float(config.get("no_gain_penalty", 0.05)),
    }


def select_action(model, obs: Dict[str, np.ndarray], device: torch.device, policy_mode: str) -> int:
    """Choose the next PPO-CE action.

    Args:
        model: Trained PPO-CE network.
        obs: Current environment observation.
        device: Target PyTorch device.
        policy_mode: Action mode, either greedy or sample.
    """

    obs_tensor = obs_to_tensor(obs, device)
    with torch.no_grad():
        logits, _ = model(obs_tensor)
        if policy_mode == "sample":
            return int(Categorical(logits=logits).sample().item())
        return int(torch.argmax(logits, dim=-1).item())


def evaluate(args):
    """Evaluate PPO-CE on a set of ESG episodes.

    Args:
        args: Parsed command-line arguments.
    """

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    runtime = resolve_runtime_config(args, checkpoint)
    label_to_idx = checkpoint.get("label_to_idx")
    if not label_to_idx:
        raise ValueError("Checkpoint does not contain label_to_idx.")

    esgs = load_esgs(args.esg_file)
    if args.limit_graphs > 0:
        esgs = esgs[: args.limit_graphs]
    env = ESGSubgraphEnv(
        esgs=esgs,
        label_to_idx=label_to_idx,
        max_nodes=runtime["max_nodes"],
        max_steps=runtime["max_steps"],
        seed=args.seed,
        terminal_cov_weight=runtime["terminal_cov_weight"],
        terminal_attr_weight=runtime["terminal_attr_weight"],
        terminal_size_penalty=runtime["terminal_size_penalty"],
        step_cov_weight=runtime["step_cov_weight"],
        step_attr_weight=runtime["step_attr_weight"],
        step_size_penalty=runtime["step_size_penalty"],
        no_gain_penalty=runtime["no_gain_penalty"],
    )

    model = ESGPolicyValueNet(
        num_labels=max(1, len(label_to_idx)),
        node_feat_dim=7,
        label_emb_dim=runtime["label_emb_dim"],
        hidden_dim=runtime["hidden_dim"],
        max_nodes=runtime["max_nodes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    episode_count = len(esgs) if args.max_episodes <= 0 else min(len(esgs), args.max_episodes)
    rows = []
    rewards = []

    for episode in range(episode_count):
        obs = env.reset(graph_idx=episode)
        done = False
        episode_reward = 0.0
        last_info = {"metrics": {}}

        while not done:
            action = select_action(model, obs, device, args.policy_mode)
            obs, reward, done, last_info = env.step(action)
            episode_reward += float(reward)

        metrics = last_info.get("metrics", {})
        row = {
            "episode": episode,
            "esg_id": last_info.get("esg_id", ""),
            "graph_idx": last_info.get("graph_idx", episode),
            "steps": last_info.get("step", 0),
            "reward": float(episode_reward),
            "coverage": float(metrics.get("coverage", 0.0)),
            "diversity": float(metrics.get("diversity", 0.0)),
            "complexity": float(metrics.get("complexity", 0.0)),
            "faithfulness": float(metrics.get("faithfulness", 0.0)),
            "selected_nodes": float(metrics.get("selected_nodes", 0.0)),
            "total_nodes": float(metrics.get("total_nodes", 0.0)),
        }
        rows.append(row)
        rewards.append(float(episode_reward))

    summary = {
        "episodes": int(len(rows)),
        "policy_mode": args.policy_mode,
        "esg_file": args.esg_file,
        "checkpoint": args.checkpoint,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "reward_min": float(np.min(rewards)) if rewards else 0.0,
        "reward_max": float(np.max(rewards)) if rewards else 0.0,
        "coverage_mean": float(np.mean([row["coverage"] for row in rows])) if rows else 0.0,
        "diversity_mean": float(np.mean([row["diversity"] for row in rows])) if rows else 0.0,
        "complexity_mean": float(np.mean([row["complexity"] for row in rows])) if rows else 0.0,
        "faithfulness_mean": float(np.mean([row["faithfulness"] for row in rows])) if rows else 0.0,
    }

    print(json.dumps(summary, indent=2))

    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["episode"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved evaluation rows: {csv_path}")

    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Saved evaluation metrics: {metrics_path}")


def parse_args():
    """Parse command-line arguments for PPO-CE evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate PPO-CE for ESG subgraph selection.")
    parser.add_argument("--esg-file", type=str, required=True, help="Path to the ESG JSON or JSONL file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved PPO-CE checkpoint.")
    parser.add_argument("--policy-mode", type=str, default="greedy", choices=["greedy", "sample"], help="Action selection mode during evaluation.")
    parser.add_argument("--max-episodes", type=int, default=0, help="Maximum number of episodes to evaluate. Use 0 to evaluate all graphs.")
    parser.add_argument("--limit-graphs", type=int, default=0, help="Optional cap on loaded ESG records before evaluation.")
    parser.add_argument("--max-nodes", type=int, default=0, help="Optional override for the environment max_nodes value.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional override for the environment max_steps value.")
    parser.add_argument("--device", type=str, default="cuda", help="Preferred execution device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--csv-out", type=str, default="", help="Optional CSV path for per-episode outputs.")
    parser.add_argument("--metrics-out", type=str, default="", help="Optional JSON path for aggregate evaluation metrics.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
