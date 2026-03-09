import argparse
import json

RELATION_MAP = {
    "left": "left_of",
    "right": "right_of",
    "standing on": "on",
    "lying on": "on",
    "attached to": "attached_to",
    "hanging on": "hanging_on",
    "supported by": "supported_by",
    "part of": "part_of",
    "build in": "built_in",
    "built in": "built_in",
    "close by": "close_by",
    "front": "in_front_of",
}

RELATION_TYPE = {
    "right_of": "spatial",
    "left_of": "spatial",
    "in_front_of": "spatial",
    "behind": "spatial",
    "above": "spatial",
    "below": "spatial",
    "on": "spatial",
    "inside": "spatial",
    "close_by": "spatial",
    "standing_on": "spatial",
    "lying_on": "spatial",
    "attached_to": "structural",
    "hanging_on": "structural",
    "supported_by": "structural",
    "built_in": "structural",
    "connected_to": "structural",
    "part_of": "part_whole",
    "same_object_type": "attribute",
    "same_color": "attribute",
    "same_shape": "attribute",
    "same_material": "attribute",
    "same_state": "attribute",
    "same_texture": "attribute",
    "same_as": "attribute",
    "same_symmetry_as": "attribute",
    "brighter_than": "comparative",
    "darker_than": "comparative",
    "bigger_than": "comparative",
    "smaller_than": "comparative",
    "higher_than": "comparative",
    "lower_than": "comparative",
    "more_open": "comparative",
    "more_closed": "comparative",
    "more_comfortable_than": "comparative",
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
    if not s:
        return "related_to"
    return RELATION_MAP.get(s, s.replace(" ", "_"))


def _relation_type(predicate):
    return RELATION_TYPE.get(predicate, "other")


def _entity_confidence(attrs):
    base = 0.55
    bonus = min(0.25, 0.05 * len(attrs))
    return round(base + bonus, 4)


def _relation_confidence(predicate):
    t = _relation_type(predicate)
    base = {
        "spatial": 0.75,
        "structural": 0.68,
        "part_whole": 0.78,
        "attribute": 0.62,
        "comparative": 0.62,
        "other": 0.60,
    }
    return round(base.get(t, 0.60), 4)


def _extract_attributes(obj):
    attrs = []
    raw_attrs = obj.get("attributes") or {}
    if isinstance(raw_attrs, dict):
        for key, value in raw_attrs.items():
            if isinstance(value, list):
                for v in value:
                    v_norm = _norm_text(v)
                    if v_norm:
                        attrs.append(v_norm)
            else:
                v_norm = _norm_text(value)
                if v_norm:
                    attrs.append(v_norm)
    elif isinstance(raw_attrs, list):
        for v in raw_attrs:
            v_norm = _norm_text(v)
            if v_norm:
                attrs.append(v_norm)

    affordances = obj.get("affordances") or []
    for v in affordances:
        v_norm = _norm_text(v)
        if v_norm:
            attrs.append(v_norm)

    return sorted(set(attrs))


def convert_3dssg(objects_data, relationships_data, min_edge_confidence=0.0, limit=0):
    objects_by_scan = {}
    for scan in objects_data.get("scans", []):
        scan_id = scan.get("scan")
        if scan_id is None:
            continue
        objects_by_scan[str(scan_id)] = scan.get("objects", [])

    relationships_by_scan = {}
    for scan in relationships_data.get("scans", []):
        scan_id = scan.get("scan")
        if scan_id is None:
            continue
        relationships_by_scan[str(scan_id)] = scan.get("relationships", [])

    esgs = []
    scan_ids = list(objects_by_scan.keys())
    if limit and limit > 0:
        scan_ids = scan_ids[:limit]

    for scan_id in scan_ids:
        objects = objects_by_scan.get(scan_id, [])
        relationships = relationships_by_scan.get(scan_id, [])

        entities = []
        entity_ids = set()
        for obj in objects:
            obj_id = str(obj.get("id"))
            if not obj_id:
                continue
            entity_ids.add(obj_id)
            attrs = _extract_attributes(obj)
            entities.append(
                {
                    "entity_id": obj_id,
                    "label": _norm_label(obj.get("label")),
                    "attributes": attrs,
                    "confidence": _entity_confidence(attrs),
                    "explanation": {
                        "source_object_id": obj_id,
                        "evidence_type": "3dssg_object",
                    },
                }
            )

        relation_map = {}
        for rel in relationships:
            if not isinstance(rel, (list, tuple)) or len(rel) < 4:
                continue
            src_id, tgt_id, rel_id, rel_name = rel[0], rel[1], rel[2], rel[3]
            src_id = str(src_id)
            tgt_id = str(tgt_id)
            if src_id not in entity_ids or tgt_id not in entity_ids:
                continue

            pred = _norm_predicate(rel_name)
            conf = _relation_confidence(pred)
            if conf < min_edge_confidence:
                continue

            key = (src_id, pred, tgt_id)
            existing = relation_map.get(key)
            record = {
                "source_entity_id": src_id,
                "predicate": pred,
                "target_entity_id": tgt_id,
                "confidence": conf,
                "explanation": {
                    "source_relation": {
                        "source": src_id,
                        "target": tgt_id,
                        "relation_id": rel_id,
                        "relation_name": rel_name,
                    },
                    "evidence_type": "3dssg_relation",
                    "dedup_count": 1,
                },
            }
            if existing is None:
                relation_map[key] = record
            else:
                existing["explanation"]["dedup_count"] += 1
                if conf > existing["confidence"]:
                    relation_map[key]["confidence"] = conf

        esgs.append(
            {
                "esg_id": scan_id,
                "entities": entities,
                "relations": list(relation_map.values()),
            }
        )

    return esgs


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3DSSG objects/relationships into ESG JSON."
    )
    parser.add_argument(
        "--objects",
        default="3DSSG/objects.json",
        help="Path to 3DSSG objects.json",
    )
    parser.add_argument(
        "--relationships",
        default="3DSSG/relationships.json",
        help="Path to 3DSSG relationships.json",
    )
    parser.add_argument(
        "--out",
        default="esg_3dssg_output.json",
        help="Output ESG JSON file",
    )
    parser.add_argument(
        "--min-edge-confidence",
        type=float,
        default=0.0,
        help="Drop relation edges below this confidence in [0,1] (default: 0.0)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Print the first N ESGs to stdout (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of scans to convert (default: 0 = all)",
    )
    args = parser.parse_args()

    with open(args.objects, "r", encoding="utf-8") as f:
        objects_data = json.load(f)
    with open(args.relationships, "r", encoding="utf-8") as f:
        relationships_data = json.load(f)

    esgs = convert_3dssg(
        objects_data,
        relationships_data,
        min_edge_confidence=max(0.0, min(1.0, args.min_edge_confidence)),
        limit=max(0, args.limit),
    )

    if args.count and args.count > 0:
        for idx, esg in enumerate(esgs[: args.count], start=1):
            print(f"\n--- ESG {idx} (scan_id={esg['esg_id']}) ---")
            print(json.dumps(esg, indent=2, ensure_ascii=False))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(esgs, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(esgs)} ESGs to: {args.out}")


if __name__ == "__main__":
    main()
