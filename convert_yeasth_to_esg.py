import argparse
import json
from collections import defaultdict
from pathlib import Path


NODE_LABELS = {
    0: "Cu",
    1: "O",
    2: "N",
    3: "C",
    4: "H",
    5: "Y",
    6: "Nd",
    7: "Pt",
    8: "P",
    9: "Sn",
    10: "Fe",
    11: "S",
    12: "Cl",
    13: "Eu",
    14: "F",
    15: "Ti",
    16: "Zr",
    17: "Hf",
    18: "Br",
    19: "Na",
    20: "Hg",
    21: "La",
    22: "Ce",
    23: "Zn",
    24: "Mn",
    25: "Co",
    26: "Ni",
    27: "I",
    28: "Au",
    29: "Pb",
    30: "Pd",
    31: "Ge",
    32: "K",
    33: "Tl",
    34: "As",
    35: "Ru",
    36: "Cd",
    37: "Ga",
    38: "Se",
    39: "Bi",
    40: "Sb",
    41: "Si",
    42: "B",
    43: "Rh",
    44: "Mo",
    45: "Nb",
    46: "In",
    47: "Os",
    48: "Ag",
    49: "Gd",
    50: "Ba",
    51: "Er",
    52: "W",
    53: "V",
    54: "Dy",
    55: "Sm",
    56: "Te",
    57: "Cr",
    58: "Mg",
    59: "Ir",
    60: "Li",
    61: "Po",
    62: "Al",
    63: "Re",
    64: "Fr",
    65: "Ta",
    66: "Cs",
    67: "Ho",
    68: "Pr",
    69: "Tb",
    70: "Ac",
    71: "Be",
    72: "Ca",
    73: "Rb",
    74: "Am",
}

EDGE_LABELS = {
    0: "bond_1",
    1: "bond_2",
    2: "bond_3",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert YeastH TU-format dataset to ESG.")
    parser.add_argument(
        "--output-file",
        default="",
        help="Optional output path. Defaults to Data/YeastH/YeastH_esg.jsonl.",
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
        help="Output format. jsonl is recommended for this dataset.",
    )
    return parser.parse_args()


def load_lines(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle]


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
    base = root / "Data" / "YeastH"

    output_file = args.output_file
    if not output_file:
        suffix = "jsonl" if args.format == "jsonl" else "json"
        output_file = str(base / f"YeastH_esg.{suffix}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph_indicator = load_lines(base / "YeastH_graph_indicator.txt")
    node_labels = load_lines(base / "YeastH_node_labels.txt")
    edge_labels = load_lines(base / "YeastH_edge_labels.txt")
    graph_labels = load_lines(base / "YeastH_graph_labels.txt")

    node_to_graph = [0]
    graph_to_nodes = defaultdict(list)
    for node_id, graph_id_str in enumerate(graph_indicator, 1):
        graph_id = int(graph_id_str)
        node_to_graph.append(graph_id)
        graph_to_nodes[graph_id].append(node_id)

    graph_edges = defaultdict(list)
    with open(base / "YeastH_A.txt", "r", encoding="utf-8") as handle:
        for edge_index, line in enumerate(handle):
            source, target = [int(value.strip()) for value in line.split(",")]
            if source > target:
                continue
            graph_id = node_to_graph[source]
            if graph_id != node_to_graph[target]:
                continue
            if args.limit_graphs > 0 and graph_id > args.limit_graphs:
                continue
            raw_edge_label = int(edge_labels[edge_index])
            graph_edges[graph_id].append((source, target, raw_edge_label))

    n_graphs = len(graph_labels) if args.limit_graphs <= 0 else min(len(graph_labels), args.limit_graphs)

    def esg_iter():
        for graph_id in range(1, n_graphs + 1):
            graph_nodes = graph_to_nodes[graph_id]
            local_id = {node_id: idx for idx, node_id in enumerate(graph_nodes)}

            entities = []
            for node_id in graph_nodes:
                raw_node_label = int(node_labels[node_id - 1])
                entities.append(
                    {
                        "entity_id": str(local_id[node_id]),
                        "label": NODE_LABELS.get(raw_node_label, f"atom_{raw_node_label}"),
                        "attributes": [],
                        "confidence": 1.0,
                        "explanation": {
                            "source_object_id": str(node_id),
                            "evidence_type": "yeasth_node",
                        },
                    }
                )

            relations = []
            for rel_index, (source, target, raw_edge_label) in enumerate(graph_edges.get(graph_id, [])):
                relations.append(
                    {
                        "source_entity_id": str(local_id[source]),
                        "predicate": EDGE_LABELS.get(raw_edge_label, f"bond_{raw_edge_label}"),
                        "target_entity_id": str(local_id[target]),
                        "confidence": 1.0,
                        "explanation": {
                            "source_relation": {
                                "source": str(source),
                                "target": str(target),
                                "edge_index": rel_index,
                                "edge_label_id": raw_edge_label,
                            },
                            "evidence_type": "yeasth_edge",
                            "dedup_count": 1,
                        },
                    }
                )

            yield {
                "esg_id": str(graph_id),
                "graph_label": {
                    "task": "binary_graph_classification",
                    "value": int(graph_labels[graph_id - 1]),
                },
                "entities": entities,
                "relations": relations,
            }

    if args.format == "jsonl":
        write_jsonl(output_path, esg_iter())
    else:
        write_json(output_path, list(esg_iter()))

    print(f"Saved {n_graphs} ESGs to {output_path}")


if __name__ == "__main__":
    main()
