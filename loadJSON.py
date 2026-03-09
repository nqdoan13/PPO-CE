import argparse
import json


RELATION_MAP = {
    "to the right of": "right_of",
    "right of": "right_of",
    "to right of": "right_of",
    "to the left of": "left_of",
    "left of": "left_of",
    "to left of": "left_of",
    "above": "above",
    "over": "above",
    "on top of": "above",
    "below": "below",
    "under": "below",
    "on": "on",
    "inside": "inside",
    "in": "inside",
    "wearing": "wearing",
    "holding": "holding",
    "has": "has",
    "of": "part_of",
}

RELATION_TYPE = {
    "right_of": "spatial",
    "left_of": "spatial",
    "above": "spatial",
    "below": "spatial",
    "on": "spatial",
    "inside": "spatial",
    "wearing": "interaction",
    "holding": "interaction",
    "has": "structural",
    "part_of": "part_whole",
}


def _norm_text(v):
    return str(v or "").strip().lower()


def _norm_label(label):
    s = _norm_text(label)
    if not s:
        return "unknown"
    if s.endswith("ies") and len(s) > 3:
        return s[:-3] + "y"
    if s.endswith("s") and not s.endswith("ss") and len(s) > 3:
        return s[:-1]
    return s


def _norm_predicate(name):
    s = _norm_text(name)
    return RELATION_MAP.get(s, s.replace(" ", "_") if s else "related_to")


def _relation_type(predicate):
    return RELATION_TYPE.get(predicate, "other")


def _entity_confidence(obj):
    attrs = obj.get("attributes") or []
    base = 0.55
    bonus = min(0.25, 0.05 * len(attrs))
    return round(base + bonus, 4)


def _relation_confidence(predicate):
    t = _relation_type(predicate)
    base = {
        "spatial": 0.75,
        "interaction": 0.70,
        "part_whole": 0.78,
        "structural": 0.68,
        "other": 0.60,
    }
    return round(base[t], 4)


def scene_graph_to_esg(scene_id, scene_graph, min_edge_confidence=0.0):
    objects = scene_graph.get("objects", {})
    entities = []
    entity_ids = set()
    attributes_non_empty = 0

    for obj_id, obj in objects.items():
        obj_id = str(obj_id)
        entity_ids.add(obj_id)
        attrs = [_norm_text(a) for a in (obj.get("attributes") or []) if _norm_text(a)]
        if attrs:
            attributes_non_empty += 1
        entity = {
            "entity_id": obj_id,
            "label": _norm_label(obj.get("name")),
            "attributes": sorted(set(attrs)),
            "confidence": _entity_confidence(obj),
            "explanation": {
                "source_object_id": obj_id,
                "evidence_type": "detected_object",
            },
        }
        entities.append(entity)

    relation_map = {}
    raw_rel_count = 0
    orphan_removed = 0
    low_conf_removed = 0

    for src_id, obj in objects.items():
        src_id = str(src_id)
        for rel in obj.get("relations", []):
            raw_rel_count += 1
            tgt_id = str(rel.get("object"))
            if src_id not in entity_ids or tgt_id not in entity_ids:
                orphan_removed += 1
                continue
            pred = _norm_predicate(rel.get("name"))
            conf = _relation_confidence(pred)
            if conf < min_edge_confidence:
                low_conf_removed += 1
                continue

            key = (src_id, pred, tgt_id)
            existing = relation_map.get(key)
            record = {
                "source_entity_id": src_id,
                "predicate": pred,
                "target_entity_id": tgt_id,
                "confidence": conf,
                "explanation": {
                    "source_object_id": src_id,
                    "source_relation": rel,
                    "evidence_type": "annotated_relation",
                    "dedup_count": 1,
                },
            }
            if existing is None:
                relation_map[key] = record
            else:
                existing["explanation"]["dedup_count"] += 1
                if conf > existing["confidence"]:
                    relation_map[key]["confidence"] = conf

    relations = list(relation_map.values())
    dedup_removed = raw_rel_count - orphan_removed - low_conf_removed - len(relations)

    connected_nodes = set()
    for r in relations:
        connected_nodes.add(r["source_entity_id"])
        connected_nodes.add(r["target_entity_id"])
    isolated_nodes = len(entity_ids - connected_nodes) if entity_ids else 0

    return {
        "esg_id": str(scene_id),
        "entities": entities,
        "relations": relations,
    }


def iter_scene_graphs(data):
    if isinstance(data, dict):
        for scene_id, scene_graph in data.items():
            yield scene_id, scene_graph
    elif isinstance(data, list):
        for idx, scene_graph in enumerate(data):
            yield idx, scene_graph
    else:
        raise ValueError("Unsupported JSON structure. Expected a dict or list.")


def print_first_esgs(data, n=3, min_edge_confidence=0.0):
    total = len(data) if isinstance(data, (list, dict)) else 0
    print(f"Total scene graphs: {total}")

    for idx, (scene_id, scene_graph) in enumerate(iter_scene_graphs(data), start=1):
        if idx > n:
            break
        esg = scene_graph_to_esg(
            scene_id, scene_graph, min_edge_confidence=min_edge_confidence
        )
        print(f"\n--- ESG {idx} (scene_id={scene_id}) ---")
        print(json.dumps(esg, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="Convert scene graphs to ESGs and print the first few results."
    )
    parser.add_argument(
        "--file",
        default="val_sceneGraphs.json",
        help="Path to the JSON file (default: val_sceneGraphs.json)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of scene graphs to print (default: 3)",
    )
    parser.add_argument(
        "--out",
        default="esg_output.json",
        help="Output JSON file path for saving ESGs (default: esg_output.json)",
    )
    parser.add_argument(
        "--min-edge-confidence",
        type=float,
        default=0.0,
        help="Drop relation edges below this confidence in [0,1] (default: 0.0)",
    )
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print_first_esgs(
        data, max(args.count, 0), min_edge_confidence=max(0.0, min(1.0, args.min_edge_confidence))
    )

    esgs = []
    for scene_id, scene_graph in iter_scene_graphs(data):
        esgs.append(
            scene_graph_to_esg(
                scene_id,
                scene_graph,
                min_edge_confidence=max(0.0, min(1.0, args.min_edge_confidence)),
            )
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(esgs, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(esgs)} ESGs to: {args.out}")


if __name__ == "__main__":
    main()
