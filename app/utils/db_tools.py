# Tools to parse SQL dump or simple JSON into internal schema structures

def load_schema_from_json(data: dict) -> dict:
    """
    Lightweight validator/normalizer for the expected schema payload.
    Ensures presence of keys and correct basic shapes.
    """
    schema = data or {}
    tables = schema.get("tables", [])
    relations = schema.get("relations", [])
    usage = schema.get("usage", [])

    normalized_tables = []
    for t in tables:
        name = t.get("name")
        cols = t.get("columns", [])
        if not name:
            continue
        normalized_tables.append({"name": name, "columns": cols})

    normalized_relations = []
    for r in relations:
        src = r.get("from") or r.get("from_table")
        dst = r.get("to") or r.get("to_table")
        rel_type = r.get("type", "")
        if not (src and dst):
            continue
        normalized_relations.append({"from": src, "to": dst, "type": rel_type})

    normalized_usage = []
    for u in usage:
        tables_pair = u.get("tables", [])
        count = u.get("count", 0)
        if len(tables_pair) == 2:
            normalized_usage.append({"tables": tables_pair, "count": int(count)})

    return {
        "tables": normalized_tables,
        "relations": normalized_relations,
        "usage": normalized_usage
    }
