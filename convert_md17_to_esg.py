import argparse
import json
from collections import defaultdict
from pathlib import Path


ATOM_LABELS = {
    0: "C",
    1: "O",
    2: "H",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MD-17 TU-format datasets such as aspirin and benzene into ESG."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["aspirin", "benzene"],
        help="Dataset name under Data/<dataset>/.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Optional output path. Defaults to Data/<dataset>/<dataset>_esg.jsonl.",
    )
    parser.add_argument(
        "--limit-graphs",
        type=int,
        default=0,
        help="Optional limit on the number of graphs to convert.",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format. jsonl is recommended for these large datasets.",
    )
    return parser.parse_args()


def dataset_paths(root: Path, dataset: str):
    base = root / "Data" / dataset
    return {
        "base": base,
        "graph_indicator": base / f"{dataset}_graph_indicator.txt",
        "node_labels": base / f"{dataset}_node_labels.txt",
        "node_attributes": base / f"{dataset}_node_attributes.txt",
        "graph_attributes": base / f"{dataset}_graph_attributes.txt",
        "edges": base / f"{dataset}_A.txt",
    }


def load_graph_indicator(path: Path):
    node_to_graph = [0]
    graph_to_nodes = defaultdict(list)
    with open(path, "r", encoding="utf-8") as handle:
        for node_id, line in enumerate(handle, 1):
            graph_id = int(line.strip())
            node_to_graph.append(graph_id)
            graph_to_nodes[graph_id].append(node_id)
    return node_to_graph, graph_to_nodes


def load_node_labels(path: Path):
    labels = [None]
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            label_id = int(line.strip())
            labels.append(ATOM_LABELS.get(label_id, f"atom_{label_id}"))
    return labels


def load_node_attributes(path: Path):
    attrs = [None]
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            values = [value.strip() for value in line.split(",")]
            if len(values) >= 6:
                attrs.append(
                    [
                        f"x:{values[0]}",
                        f"y:{values[1]}",
                        f"z:{values[2]}",
                        f"fx:{values[3]}",
                        f"fy:{values[4]}",
                        f"fz:{values[5]}",
                    ]
                )
            else:
                attrs.append([])
    return attrs


def load_graph_attributes(path: Path):
    graph_attrs = {}
    with open(path, "r", encoding="utf-8") as handle:
        for graph_id, line in enumerate(handle, 1):
            graph_attrs[graph_id] = float(line.strip())
    return graph_attrs


def load_graph_edges(path: Path, node_to_graph, limit_graphs: int):
    graph_edges = defaultdict(list)
    seen = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            source, target = [int(value.strip()) for value in line.split(",")]
            if source > target:
                continue
            graph_id = node_to_graph[source]
            if graph_id != node_to_graph[target]:
                continue
            if limit_graphs > 0 and graph_id > limit_graphs:
                continue
            key = (graph_id, source, target)
            if key in seen:
                continue
            seen.add(key)
            graph_edges[graph_id].append((source, target))
    return graph_edges


def build_esg(graph_id, graph_nodes, graph_edges, node_labels, node_attributes, graph_energy):
    local_id = {node_id: idx for idx, node_id in enumerate(graph_nodes)}
    entities = []
    for node_id in graph_nodes:
        entities.append(
            {
                "entity_id": str(local_id[node_id]),
                "label": node_labels[node_id],
                "attributes": node_attributes[node_id] or [],
                "confidence": 1.0,
                "explanation": {
                    "source_object_id": str(node_id),
                    "evidence_type": "md17_atom",
                },
            }
        )

    relations = []
    for edge_index, (source, target) in enumerate(graph_edges):
        relations.append(
            {
                "source_entity_id": str(local_id[source]),
                "predicate": "within_5a",
                "target_entity_id": str(local_id[target]),
                "confidence": 1.0,
                "explanation": {
                    "source_relation": {
                        "source": str(source),
                        "target": str(target),
                        "edge_index": edge_index,
                    },
                    "evidence_type": "md17_spatial_edge",
                    "dedup_count": 1,
                },
            }
        )

    return {
        "esg_id": str(graph_id),
        "graph_label": {
            "task": "energy_regression",
            "value": graph_energy,
        },
        "entities": entities,
        "relations": relations,
    }


def write_jsonl(output_path: Path, esgs):
    with open(output_path, "w", encoding="utf-8") as handle:
        for esg in esgs:
            handle.write(json.dumps(esg, ensure_ascii=False) + "\n")


def write_json(output_path: Path, esgs):
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(esgs, handle, ensure_ascii=False)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    paths = dataset_paths(root, args.dataset)

    output_file = args.output_file
    if not output_file:
        suffix = "jsonl" if args.format == "jsonl" else "json"
        output_file = str(paths["base"] / f"{args.dataset}_esg.{suffix}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    node_to_graph, graph_to_nodes = load_graph_indicator(paths["graph_indicator"])
    node_labels = load_node_labels(paths["node_labels"])
    node_attributes = load_node_attributes(paths["node_attributes"])
    graph_attributes = load_graph_attributes(paths["graph_attributes"])
    graph_edges = load_graph_edges(paths["edges"], node_to_graph, args.limit_graphs)

    max_graph_id = max(graph_to_nodes) if graph_to_nodes else 0
    n_graphs = min(max_graph_id, args.limit_graphs) if args.limit_graphs > 0 else max_graph_id

    def esg_iter():
        for graph_id in range(1, n_graphs + 1):
            yield build_esg(
                graph_id=graph_id,
                graph_nodes=graph_to_nodes[graph_id],
                graph_edges=graph_edges.get(graph_id, []),
                node_labels=node_labels,
                node_attributes=node_attributes,
                graph_energy=graph_attributes[graph_id],
            )

    if args.format == "jsonl":
        write_jsonl(output_path, esg_iter())
    else:
        write_json(output_path, list(esg_iter()))

    print(f"Saved {n_graphs} ESGs to {output_path}")


if __name__ == "__main__":
    main()
