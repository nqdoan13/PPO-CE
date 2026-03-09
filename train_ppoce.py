import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from esg_env import ESGSubgraphEnv
from model import ESGPolicyValueNet


@dataclass
class Transition:
    obs: Dict[str, np.ndarray]
    action: int
    reward: float
    done: float
    logp: float
    value: float


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_esgs(path: str, limit_graphs: int) -> List[Dict]:
    if path.lower().endswith(".jsonl"):
        esgs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                esgs.append(json.loads(line))
                if limit_graphs > 0 and len(esgs) >= limit_graphs:
                    break
    else:
        with open(path, "r", encoding="utf-8") as f:
            esgs = json.load(f)
        if not isinstance(esgs, list):
            raise ValueError("Expected ESG JSON file as a list.")
    if limit_graphs > 0:
        esgs = esgs[:limit_graphs]
    if not esgs:
        raise ValueError("No ESG entries found after applying --limit-graphs.")
    return esgs


def build_label_vocab(esgs: List[Dict], max_labels: int = 2000) -> Dict[str, int]:
    counts = Counter()
    for g in esgs:
        for e in g.get("entities", []):
            label = str(e.get("label") or "unknown")
            counts[label] += 1
    most_common = counts.most_common(max_labels)
    return {label: i + 1 for i, (label, _) in enumerate(most_common)}


def obs_to_tensor(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "node_feats": torch.tensor(obs["node_feats"], dtype=torch.float32, device=device).unsqueeze(0),
        "label_ids": torch.tensor(obs["label_ids"], dtype=torch.long, device=device).unsqueeze(0),
        "selected_mask": torch.tensor(obs["selected_mask"], dtype=torch.float32, device=device).unsqueeze(0),
        "valid_mask": torch.tensor(obs["valid_mask"], dtype=torch.float32, device=device).unsqueeze(0),
        "action_mask": torch.tensor(obs["action_mask"], dtype=torch.float32, device=device).unsqueeze(0),
    }


def batch_obs_to_tensor(obs_batch: List[Dict[str, np.ndarray]], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "node_feats": torch.tensor(np.stack([o["node_feats"] for o in obs_batch]), dtype=torch.float32, device=device),
        "label_ids": torch.tensor(np.stack([o["label_ids"] for o in obs_batch]), dtype=torch.long, device=device),
        "selected_mask": torch.tensor(
            np.stack([o["selected_mask"] for o in obs_batch]), dtype=torch.float32, device=device
        ),
        "valid_mask": torch.tensor(np.stack([o["valid_mask"] for o in obs_batch]), dtype=torch.float32, device=device),
        "action_mask": torch.tensor(
            np.stack([o["action_mask"] for o in obs_batch]), dtype=torch.float32, device=device
        ),
    }


def compute_gae(
    transitions: List[Transition],
    last_value: float,
    gamma: float,
    gae_lambda: float,
):
    advantages = []
    gae = 0.0
    for t in reversed(range(len(transitions))):
        next_non_terminal = 1.0 - transitions[t].done
        next_value = last_value if t == len(transitions) - 1 else transitions[t + 1].value
        delta = transitions[t].reward + gamma * next_value * next_non_terminal - transitions[t].value
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.append(gae)
    advantages.reverse()
    returns = [adv + tr.value for adv, tr in zip(advantages, transitions)]
    return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)


def ppoce_train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    esgs = load_esgs(args.esg_file, args.limit_graphs)
    label_to_idx = build_label_vocab(esgs, max_labels=args.max_labels)
    print(f"Loaded {len(esgs)} ESGs, label vocab size={len(label_to_idx)}")

    env = ESGSubgraphEnv(
        esgs=esgs,
        label_to_idx=label_to_idx,
        max_nodes=args.max_nodes,
        max_steps=args.max_steps,
        seed=args.seed,
        terminal_cov_weight=args.terminal_cov_weight,
        terminal_attr_weight=args.terminal_attr_weight,
        terminal_size_penalty=args.terminal_size_penalty,
        step_cov_weight=args.step_cov_weight,
        step_attr_weight=args.step_attr_weight,
        step_size_penalty=args.step_size_penalty,
        no_gain_penalty=args.no_gain_penalty,
    )
    model = ESGPolicyValueNet(
        num_labels=max(1, len(label_to_idx)),
        node_feat_dim=7,
        label_emb_dim=args.label_emb_dim,
        hidden_dim=args.hidden_dim,
        max_nodes=args.max_nodes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    obs = env.reset()
    episode_rewards = []
    running_ep_reward = 0.0
    stopped_early = False

    for update in range(1, args.updates + 1):
        rollout: List[Transition] = []
        for _ in range(args.rollout_steps):
            obs_t = obs_to_tensor(obs, device)
            with torch.no_grad():
                logits, value = model(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample().item()
                logp = dist.log_prob(torch.tensor([action], device=device)).item()

            next_obs, reward, done, info = env.step(action)
            if args.print_step_metrics:
                m = info.get("metrics", {})
                print(
                    f"step graph={info.get('graph_idx')} ep_steps={info.get('step')} "
                    f"action={action} reward={float(reward):.4f} done={bool(done)} "
                    f"Cov={float(m.get('coverage', 0.0)):.4f} "
                    f"Div={float(m.get('diversity', 0.0)):.4f} "
                    f"Faith={float(m.get('faithfulness', 0.0)):.4f} "
                    f"Complex={float(m.get('complexity', 0.0)):.4f}"
                )
            rollout.append(
                Transition(
                    obs=obs,
                    action=action,
                    reward=float(reward),
                    done=float(done),
                    logp=float(logp),
                    value=float(value.item()),
                )
            )

            running_ep_reward += reward
            obs = next_obs
            if done:
                episode_rewards.append(running_ep_reward)
                running_ep_reward = 0.0
                if args.target_episodes > 0 and len(episode_rewards) >= args.target_episodes:
                    stopped_early = True
                    break
                obs = env.reset()

        if stopped_early:
            if running_ep_reward != 0.0:
                running_ep_reward = 0.0
            if not rollout:
                break

        with torch.no_grad():
            last_value = model(obs_to_tensor(obs, device))[1].item()
        advantages, returns = compute_gae(
            rollout,
            last_value=last_value,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        obs_batch = [tr.obs for tr in rollout]
        actions = torch.tensor([tr.action for tr in rollout], dtype=torch.long, device=device)
        old_logp = torch.tensor([tr.logp for tr in rollout], dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)

        n = len(rollout)
        idxs = np.arange(n)
        for _epoch in range(args.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, args.minibatch_size):
                mb = idxs[start : start + args.minibatch_size]
                mb_obs = batch_obs_to_tensor([obs_batch[i] for i in mb], device)
                mb_actions = actions[mb]
                mb_old_logp = old_logp[mb]
                mb_returns = returns_t[mb]
                mb_adv = adv_t[mb]

                logits, values = model(mb_obs)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, mb_returns)
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        recent = episode_rewards[-20:] if episode_rewards else [0.0]
        avg_ep_reward = float(np.mean(recent))
        print(
            f"update={update}/{args.updates} "
            f"episodes={len(episode_rewards)} "
            f"avg_ep_reward(last20)={avg_ep_reward:.4f}"
        )

        if stopped_early:
            print(f"Stopping early after reaching target episodes={args.target_episodes}")
            break

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ppo_esg.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_idx": label_to_idx,
            "config": vars(args),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")

    if args.rewards_out:
        rewards_path = Path(args.rewards_out)
        rewards_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rewards_path, "w", encoding="utf-8") as f:
            json.dump([float(r) for r in episode_rewards], f, indent=2)
        print(f"Saved episode rewards: {rewards_path}")

    if args.metrics_out:
        metrics = {
            "episodes": int(len(episode_rewards)),
            "reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "reward_std": float(np.std(episode_rewards)) if episode_rewards else 0.0,
            "reward_min": float(np.min(episode_rewards)) if episode_rewards else 0.0,
            "reward_max": float(np.max(episode_rewards)) if episode_rewards else 0.0,
            "reward_median": float(np.median(episode_rewards)) if episode_rewards else 0.0,
            "reward_mean_last20": float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0,
            "reward_mean_last50": float(np.mean(episode_rewards[-50:])) if episode_rewards else 0.0,
            "target_episodes": int(args.target_episodes),
            "limit_graphs": int(args.limit_graphs),
            "esg_file": args.esg_file,
            "checkpoint_path": str(ckpt_path),
        }
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved training metrics: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO-CE to select ESG subgraphs.")
    parser.add_argument("--esg-file", type=str, default="esg_output.json")
    parser.add_argument("--limit-graphs", type=int, default=2000)
    parser.add_argument("--max-labels", type=int, default=2000)
    parser.add_argument("--max-nodes", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--terminal-cov-weight", type=float, default=1.5)
    parser.add_argument("--terminal-attr-weight", type=float, default=0.5)
    parser.add_argument("--terminal-size-penalty", type=float, default=0.5)
    parser.add_argument("--step-cov-weight", type=float, default=1.0)
    parser.add_argument("--step-attr-weight", type=float, default=0.5)
    parser.add_argument("--step-size-penalty", type=float, default=0.2)
    parser.add_argument("--no-gain-penalty", type=float, default=0.05)

    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--label-emb-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--target-episodes",
        type=int,
        default=0,
        help="Stop training after this many completed episodes (0 = disabled).",
    )
    parser.add_argument(
        "--rewards-out",
        type=str,
        default="",
        help="Optional JSON path to save all episode rewards.",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="",
        help="Optional JSON path to save summary training metrics.",
    )
    parser.add_argument(
        "--print-step-metrics",
        action="store_true",
        help="Print per-step Cov/Div/Faithfulness/Complexity during rollouts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    ppoce_train(parse_args())
