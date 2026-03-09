import random
from typing import Dict, List, Optional, Tuple

import numpy as np


class ESGSubgraphEnv:
    """Environment for PPO-CE-based ESG subgraph selection.

    Args:
        esgs: ESG records used as episodes.
        label_to_idx: Label vocabulary for entity labels.
        max_nodes: Maximum number of nodes exposed per episode.
        max_steps: Maximum number of actions before forced termination.
        seed: Random seed for episode sampling.
        terminal_cov_weight: Terminal reward weight for edge coverage.
        terminal_attr_weight: Terminal reward weight for attribute diversity.
        terminal_size_penalty: Terminal reward penalty for large subgraphs.
        step_cov_weight: Step reward weight for coverage gain.
        step_attr_weight: Step reward weight for diversity gain.
        step_size_penalty: Step reward penalty for each selected node.
        no_gain_penalty: Penalty when an action gives no semantic gain.
    """

    def __init__(
        self,
        esgs: List[Dict],
        label_to_idx: Dict[str, int],
        max_nodes: int = 32,
        max_steps: int = 8,
        seed: int = 42,
        terminal_cov_weight: float = 1.5,
        terminal_attr_weight: float = 0.5,
        terminal_size_penalty: float = 0.5,
        step_cov_weight: float = 1.0,
        step_attr_weight: float = 0.5,
        step_size_penalty: float = 0.2,
        no_gain_penalty: float = 0.05,
    ):
        if not esgs:
            raise ValueError("Empty ESG list.")
        self.esgs = esgs
        self.label_to_idx = label_to_idx
        self.max_nodes = max_nodes
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.terminal_cov_weight = float(terminal_cov_weight)
        self.terminal_attr_weight = float(terminal_attr_weight)
        self.terminal_size_penalty = float(terminal_size_penalty)
        self.step_cov_weight = float(step_cov_weight)
        self.step_attr_weight = float(step_attr_weight)
        self.step_size_penalty = float(step_size_penalty)
        self.no_gain_penalty = float(no_gain_penalty)
        self.stop_action = self.max_nodes
        self.current = None
        self.selected = None
        self.steps = 0
        self.last_graph_idx = -1
        self.prev_coverage = 0.0
        self.prev_attr_div = 0.0

    def _build_graph_cache(self, esg: Dict) -> Dict:
        """Convert one ESG into fixed-size tensors and cached statistics.

        Args:
            esg: ESG record for the current episode.
        """

        entities = esg.get("entities", [])
        relations = esg.get("relations", [])
        n = min(len(entities), self.max_nodes)

        entity_ids = [str(entity.get("entity_id")) for entity in entities[:n]]
        id_to_local = {entity_id: i for i, entity_id in enumerate(entity_ids)}

        in_deg = np.zeros(n, dtype=np.float32)
        out_deg = np.zeros(n, dtype=np.float32)
        in_rel_conf = np.zeros(n, dtype=np.float32)
        out_rel_conf = np.zeros(n, dtype=np.float32)
        edge_pairs: List[Tuple[int, int]] = []

        for relation in relations:
            source = id_to_local.get(str(relation.get("source_entity_id")))
            target = id_to_local.get(str(relation.get("target_entity_id")))
            if source is None or target is None:
                continue
            edge_pairs.append((source, target))
            out_deg[source] += 1.0
            in_deg[target] += 1.0
            rel_conf = float(relation.get("confidence") or 0.0)
            out_rel_conf[source] += rel_conf
            in_rel_conf[target] += rel_conf

        feats = np.zeros((self.max_nodes, 7), dtype=np.float32)
        label_ids = np.zeros(self.max_nodes, dtype=np.int64)
        valid_mask = np.zeros(self.max_nodes, dtype=np.bool_)
        node_scores = np.zeros(self.max_nodes, dtype=np.float32)
        unique_attrs_all = set()

        max_deg = max(1.0, float((in_deg + out_deg).max()) if n > 0 else 1.0)
        for i, entity in enumerate(entities[:n]):
            attrs = entity.get("attributes") or []
            for attr in attrs:
                unique_attrs_all.add(str(attr))
            attr_count = min(8.0, float(len(attrs))) / 8.0
            deg = float(in_deg[i] + out_deg[i]) / max_deg
            node_conf = float(entity.get("confidence") or 0.0)
            in_rel_conf_mean = in_rel_conf[i] / max(1.0, in_deg[i])
            out_rel_conf_mean = out_rel_conf[i] / max(1.0, out_deg[i])
            feats[i] = np.array(
                [
                    attr_count,
                    in_deg[i] / max_deg,
                    out_deg[i] / max_deg,
                    deg,
                    node_conf,
                    in_rel_conf_mean,
                    out_rel_conf_mean,
                ]
            )
            label = str(entity.get("label") or "unknown")
            label_ids[i] = self.label_to_idx.get(label, 0)
            valid_mask[i] = True
            node_scores[i] = 0.6 * deg + 0.3 * attr_count + 0.1 * node_conf

        return {
            "esg_id": str(esg.get("esg_id", "")),
            "feats": feats,
            "label_ids": label_ids,
            "valid_mask": valid_mask,
            "edge_pairs": edge_pairs,
            "node_scores": node_scores,
            "n_nodes": int(valid_mask.sum()),
            "unique_attrs_all": unique_attrs_all,
            "entities": entities[:n],
        }

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Build the observation dictionary for the current state."""

        action_mask = self.current["valid_mask"] & (~self.selected)
        stop_valid = np.array([True], dtype=np.bool_)
        action_mask = np.concatenate([action_mask, stop_valid], axis=0)
        return {
            "node_feats": self.current["feats"].copy(),
            "label_ids": self.current["label_ids"].copy(),
            "selected_mask": self.selected.astype(np.float32).copy(),
            "valid_mask": self.current["valid_mask"].astype(np.float32).copy(),
            "action_mask": action_mask.astype(np.float32),
        }

    def reset(self, graph_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Start a new episode.

        Args:
            graph_idx: Optional ESG index to evaluate deterministically.
        """

        if graph_idx is None:
            graph_idx = self.rng.randrange(len(self.esgs))
        graph_idx = int(graph_idx) % len(self.esgs)
        self.last_graph_idx = graph_idx
        self.current = self._build_graph_cache(self.esgs[graph_idx])
        self.selected = np.zeros(self.max_nodes, dtype=np.bool_)
        self.steps = 0
        self.prev_coverage = 0.0
        self.prev_attr_div = 0.0
        return self._get_obs()

    def _current_stats(self):
        """Compute coverage, diversity, size, and selected indices."""

        n_nodes = self.current["n_nodes"]
        selected_idx = set(np.where(self.selected)[0].tolist())
        edges = self.current["edge_pairs"]
        if edges:
            covered = sum(1 for source, target in edges if source in selected_idx and target in selected_idx)
            coverage = covered / max(1, len(edges))
        else:
            coverage = 0.0

        selected_attrs = set()
        for idx in selected_idx:
            attrs = self.current["entities"][idx].get("attributes") or []
            for attr in attrs:
                selected_attrs.add(str(attr))
        all_attrs = self.current["unique_attrs_all"]
        attr_div = len(selected_attrs) / max(1, len(all_attrs)) if all_attrs else 0.0
        size_ratio = len(selected_idx) / max(1, n_nodes)
        return coverage, attr_div, size_ratio, selected_idx

    def _semantic_utility(self, selected_idx: set) -> float:
        """Score a selected node set without the size penalty.

        Args:
            selected_idx: Selected node indices.
        """

        edges = self.current["edge_pairs"]
        if edges:
            covered = sum(1 for source, target in edges if source in selected_idx and target in selected_idx)
            coverage = covered / max(1, len(edges))
        else:
            coverage = 0.0

        selected_attrs = set()
        for idx in selected_idx:
            attrs = self.current["entities"][idx].get("attributes") or []
            for attr in attrs:
                selected_attrs.add(str(attr))
        all_attrs = self.current["unique_attrs_all"]
        attr_div = len(selected_attrs) / max(1, len(all_attrs)) if all_attrs else 0.0
        return float(self.terminal_cov_weight * coverage + self.terminal_attr_weight * attr_div)

    def _step_metrics(self) -> Dict[str, float]:
        """Return episode metrics for the current state."""

        coverage, attr_div, size_ratio, selected_idx = self._current_stats()
        n_nodes = int(self.current["n_nodes"])
        full_set = set(range(n_nodes))
        full_u = self._semantic_utility(full_set) if n_nodes > 0 else 0.0
        selected_u = self._semantic_utility(selected_idx) if selected_idx else 0.0
        faithfulness = (selected_u / full_u) if full_u > 1e-8 else 0.0
        return {
            "coverage": float(coverage),
            "diversity": float(attr_div),
            "complexity": float(size_ratio),
            "faithfulness": float(faithfulness),
            "selected_nodes": float(len(selected_idx)),
            "total_nodes": float(n_nodes),
        }

    def _terminal_reward(self) -> float:
        """Compute the terminal reward for the selected subgraph."""

        n_nodes = self.current["n_nodes"]
        if n_nodes == 0:
            return -0.5
        coverage, attr_div, size_ratio, selected_idx = self._current_stats()
        if not selected_idx:
            return -0.4
        return float(
            self.terminal_cov_weight * coverage
            + self.terminal_attr_weight * attr_div
            - self.terminal_size_penalty * size_ratio
        )

    def step(self, action: int):
        """Advance the environment by one action.

        Args:
            action: Node index to select or the stop action.
        """

        if self.current is None:
            raise RuntimeError("Call reset() before step().")

        self.steps += 1
        done = False
        reward = 0.0
        info = {"esg_id": self.current["esg_id"], "graph_idx": self.last_graph_idx}

        if action == self.stop_action:
            reward += self._terminal_reward()
            done = True
            info["metrics"] = self._step_metrics()
            info["step"] = int(self.steps)
            return self._get_obs(), reward, done, info

        if action < 0 or action >= self.max_nodes:
            reward -= 0.2
        elif not self.current["valid_mask"][action]:
            reward -= 0.2
        elif self.selected[action]:
            reward -= 0.1
        else:
            self.selected[action] = True
            coverage, attr_div, _, _ = self._current_stats()
            delta_cov = max(0.0, coverage - self.prev_coverage)
            delta_attr = max(0.0, attr_div - self.prev_attr_div)
            delta_size = 1.0 / max(1, self.current["n_nodes"])
            reward += float(
                self.step_cov_weight * delta_cov
                + self.step_attr_weight * delta_attr
                - self.step_size_penalty * delta_size
            )
            if delta_cov <= 1e-8 and delta_attr <= 1e-8:
                reward -= self.no_gain_penalty
            self.prev_coverage = coverage
            self.prev_attr_div = attr_div

        if self.steps >= self.max_steps:
            reward += self._terminal_reward()
            done = True

        info["metrics"] = self._step_metrics()
        info["step"] = int(self.steps)
        return self._get_obs(), reward, done, info
