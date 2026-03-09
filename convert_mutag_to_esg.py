import argparse
import json
import os
from pathlib import Path


ATOM_LABELS = {
    0: "C",
    1: "N",
    2: "O",
    3: "F",
    4: "I",
    5: "Cl",
    6: "Br",
}

BOND_LABELS = {
    0: "single_bond",
    1: "double_bond",
    2: "triple_bond",
    3: "aromatic_bond",
}

DEFAULT_ROOT = os.path.join(os.path.dirname(__file__), "Data", "TUDataset")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "Data", "MUTAG", "mutag_esg.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download MUTAG via PyTorch Geometric and convert it to ESG JSON."
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="Root directory for the PyG TUDataset cache.",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT,
        help="Path to write the ESG list JSON.",
    )
    parser.add_argument(
        "--limit-graphs",
        type=int,
        default=0,
        help="Optional limit on the number of MUTAG graphs to convert.",
    )
    return parser.parse_args()


def require_pyg():
    try:
        from torch_geometric.datasets import TUDataset
    except ImportError as exc:
        raise SystemExit(
            "torch_geometric is not installed. Install it first, then rerun this script."
        ) from exc
    return TUDataset


def argmax_label(values, label_map, unknown_prefix):
    if values is None:
        return unknown_prefix

    if hasattr(values, "tolist"):
        values = values.tolist()

    if not isinstance(values, list) or not values:
        return unknown_prefix

    if len(values) == 1:
        try:
            index = int(values[0])
            return label_map.get(index, f"{unknown_prefix}_{index}")
        except (TypeError, ValueError):
            return unknown_prefix

    best_index = max(range(len(values)), key=lambda idx: values[idx])
    return label_map.get(best_index, f"{unknown_prefix}_{best_index}")


def graph_target_label(graph_y):
    if hasattr(graph_y, "view"):
        graph_y = graph_y.view(-1).tolist()
    elif hasattr(graph_y, "tolist"):
        graph_y = graph_y.tolist()

    if isinstance(graph_y, list) and graph_y:
        value = int(graph_y[0])
    else:
        value = int(graph_y)

    return "mutagenic" if value == 1 else "non_mutagenic"


def convert_graph(data, graph_index):
    entities = []
    x_rows = data.x.tolist() if getattr(data, "x", None) is not None else None
    num_nodes = int(data.num_nodes)

    for node_idx in range(num_nodes):
        node_features = x_rows[node_idx] if x_rows is not None else None
        atom_label = argmax_label(node_features, ATOM_LABELS, "atom")
        attributes = []
        if node_features is not None:
            attributes.append(f"feature_dim:{len(node_features)}")

        entities.append(
            {
                "entity_id": str(node_idx),
                "label": atom_label,
                "attributes": attributes,
                "confidence": 1.0,
                "explanation": {
                    "source_object_id": str(node_idx),
                    "evidence_type": "mutag_atom",
                },
            }
        )

    relations = []
    edge_index = data.edge_index.t().tolist()
    edge_attr_rows = data.edge_attr.tolist() if getattr(data, "edge_attr", None) is not None else None
    seen_edges = set()

    for edge_idx, pair in enumerate(edge_index):
        source, target = int(pair[0]), int(pair[1])
        edge_key = tuple(sorted((source, target)))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        bond_features = edge_attr_rows[edge_idx] if edge_attr_rows is not None else None
        predicate = argmax_label(bond_features, BOND_LABELS, "bond")
        relations.append(
            {
                "source_entity_id": str(source),
                "predicate": predicate,
                "target_entity_id": str(target),
                "confidence": 1.0,
                "explanation": {
                    "source_relation": {
                        "source": str(source),
                        "target": str(target),
                        "edge_index": edge_idx,
                    },
                    "evidence_type": "mutag_bond",
                    "dedup_count": 1,
                },
            }
        )

    return {
        "esg_id": f"MUTAG_{graph_index}",
        "graph_label": {
            "task": "mutagenicity",
            "value": int(data.y.view(-1)[0].item()),
            "label": graph_target_label(data.y),
        },
        "entities": entities,
        "relations": relations,
    }


def main():
    args = parse_args()
    TUDataset = require_pyg()

    dataset = TUDataset(root=args.root, name="MUTAG")
    limit = args.limit_graphs if args.limit_graphs > 0 else len(dataset)

    esgs = []
    for graph_index in range(min(limit, len(dataset))):
        esgs.append(convert_graph(dataset[graph_index], graph_index))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(esgs, handle, indent=2, ensure_ascii=False)

    print(f"Saved {len(esgs)} ESGs to {output_path}")


if __name__ == "__main__":
    main()
