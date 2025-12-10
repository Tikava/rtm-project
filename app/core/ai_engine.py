# backend/app/core/ai_engine.py
from typing import Dict, List, Any
from collections import defaultdict
import math
from copy import deepcopy
from statistics import median
from app.core.llm_adapter import summarize_domains_with_llm

"""
Schema format expected (simple JSON):
{
  "tables": [
    {"name": "users", "columns": ["id","name","email"]},
    {"name": "orders", "columns": ["id","user_id","total"]},
    {"name": "order_items", "columns": ["id","order_id","product_id","qty"]},
    {"name": "products", "columns": ["id","name","price"]}
  ],
  "relations": [
    {"from": "orders", "to": "users", "type": "M:1"},
    {"from": "order_items", "to": "orders", "type": "M:1"},
    {"from": "order_items", "to": "products", "type": "M:1"}
  ],
  "usage": [   # optional: sample query co-occurrence counts
    {"tables": ["orders","order_items"], "count": 120},
    {"tables": ["orders","users"], "count": 200}
  ]
}
"""

def build_adjacency(schema: Dict[str, Any]) -> Dict[str, List[str]]:
    adj = defaultdict(list)
    for r in schema.get("relations", []):
        a = r.get("from")
        b = r.get("to")
        if a and b:
            adj[a].append(b)
            adj[b].append(a)  # undirected view for clustering
    # ensure every table present
    for t in schema.get("tables", []):
        adj.setdefault(t["name"], [])
    return dict(adj)

# ---------------------
# Baseline algorithm
# ---------------------
def baseline_analyze(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very simple baseline:
    - group tables by connected components of FK graph
    - each component = one domain
    """
    adj = build_adjacency(schema)
    visited = set()
    components = []
    for node in adj:
        if node in visited:
            continue
        # BFS
        stack = [node]
        comp = []
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.append(n)
            for nb in adj.get(n, []):
                if nb not in visited:
                    stack.append(nb)
        components.append(sorted(comp))
    # compute simple metrics
    metrics = compute_metrics(schema, components)
    return {"domains": components, "metrics": metrics, "method": "baseline"}

# ---------------------
# AI (rule-based heuristics) algorithm
# ---------------------
def ai_analyze(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rule-based heuristic decomposition:
    - start from FK connectivity (like baseline)
    - apply heuristics:
      * tables with 'user', 'account', 'auth' -> user-centric domain
      * join tables (many-to-many) considered bridges; try to put them with the domain they touch most frequently (usage)
      * separate small utility tables (lookup) unless heavily used
      * prefer merging components if usage co-occurrence is high
    """
    adj = build_adjacency(schema)
    name_index = {t["name"]: t for t in schema.get("tables", [])}

    # start with connected components
    baseline = baseline_analyze(schema)
    components = baseline["domains"]

    # compute usage co-occurrence map and median for thresholds
    usage_map = defaultdict(int)
    usage_counts = []
    for u in schema.get("usage", []):
        tables = tuple(sorted(u.get("tables", [])))
        cnt = int(u.get("count", 0))
        usage_map[tables] += cnt
        usage_counts.append(cnt)
    usage_threshold = median(usage_counts) if usage_counts else 50

    # refine large components: split by usage strength and keyword seeds
    components = refine_components(components, schema, usage_map, usage_threshold)
    components = attach_small_components(components, schema, usage_map, min_size=2)

    # Merge components if heavy co-usage between them
    merged = []
    used = [False]*len(components)
    for i, comp_i in enumerate(components):
        if used[i]:
            continue
        group = set(comp_i)
        for j in range(i+1, len(components)):
            if used[j]:
                continue
            comp_j = components[j]
            # compute total usage between comp_i and comp_j
            tot = 0
            for a in comp_i:
                for b in comp_j:
                    tot += usage_map.get(tuple(sorted((a,b))), 0)
            # heuristic: merge только если сильная ко-используемость и хотя бы один компонент маленький
            if tot >= usage_threshold and (len(comp_i) <= 5 or len(comp_j) <= 5):
                group.update(comp_j)
                used[j] = True
        used[i] = True
        merged.append(sorted(group))

    # Post-process: move join tables and utility tables
    merged = post_process_join_tables(merged, schema)
    merged = attach_utility_tables(merged, schema, usage_map)

    metrics = compute_metrics(schema, merged)
    summaries = summarize_domains(merged, schema)
    llm = summarize_domains_with_llm(merged, summaries, schema)
    notes = []
    if usage_threshold:
        notes.append(f"Usage merge threshold: {usage_threshold}")
    if llm.get("note"):
        notes.append(llm["note"])

    return {
        "domains": merged,
        "domain_summaries": summaries,
        "llm_summaries": llm,
        "metrics": metrics,
        "method": "ai_rule_based",
        "notes": notes
    }

def post_process_join_tables(components: List[List[str]], schema: Dict[str, Any]) -> List[List[str]]:
    # Simple rule: if a table name contains '_' or 'link' or 'join' treat as join table
    join_candidates = set()
    for t in schema.get("tables", []):
        name = t["name"]
        low = name.lower()
        if "_" in low or "link" in low or "join" in low or "map" in low:
            # heuristically likely to be join table
            join_candidates.add(name)
    # assign each join table to the component that contains most of its FKs
    # build fast lookup
    comp_index = {}
    for idx, comp in enumerate(components):
        for t in comp:
            comp_index[t] = idx
    # get relations
    relations = schema.get("relations", [])
    add_to = defaultdict(lambda: defaultdict(int))  # join_table -> comp_idx -> count
    for r in relations:
        a = r.get("from"); b = r.get("to")
        if a in join_candidates:
            if b in comp_index:
                add_to[a][comp_index[b]] += 1
        if b in join_candidates:
            if a in comp_index:
                add_to[b][comp_index[a]] += 1
    # move join tables into best comp
    for join_table, comp_counts in add_to.items():
        if not comp_counts:
            continue
        best_comp = max(comp_counts.items(), key=lambda x: x[1])[0]
        # remove join_table from any other comp
        for comp in components:
            if join_table in comp:
                comp.remove(join_table)
        # add to best comp (if not present)
        if join_table not in components[best_comp]:
            components[best_comp].append(join_table)
            components[best_comp].sort()
    # Remove empty components
    components = [c for c in components if c]
    return components

def attach_utility_tables(components: List[List[str]], schema: Dict[str, Any], usage_map: Dict[tuple, int]) -> List[List[str]]:
    """
    Utility tables are small lookups like status/type/category. Attach them to the neighbor
    with highest usage or FK ties.
    """
    utility_keywords = ("ref", "lookup", "dict", "type", "status", "code")
    comp_index = {t: idx for idx, comp in enumerate(components) for t in comp}

    # build FK degrees
    relations = schema.get("relations", [])
    fk_adj = defaultdict(list)
    for r in relations:
        a, b = r.get("from"), r.get("to")
        if a and b:
            fk_adj[a].append(b)
            fk_adj[b].append(a)

    for table in list(comp_index.keys()):
        low = table.lower()
        if not any(k in low for k in utility_keywords):
            continue
        # find best neighbor comp by usage and degree
        counts = defaultdict(int)
        for nb in fk_adj.get(table, []):
            nb_comp = comp_index.get(nb)
            if nb_comp is None:
                continue
            counts[nb_comp] += 1
            counts[nb_comp] += usage_map.get(tuple(sorted((table, nb))), 0)
        if not counts:
            continue
        best_comp = max(counts.items(), key=lambda x: x[1])[0]
        current_comp = comp_index.get(table)
        if current_comp == best_comp:
            continue
        # move table
        for comp in components:
            if table in comp:
                comp.remove(table)
        components[best_comp].append(table)
        components[best_comp].sort()
    components = [c for c in components if c]
    return components

def summarize_domains(components: List[List[str]], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Produce lightweight human-readable summaries per domain to explain grouping.
    """
    name_hints = [
        ("user", "Users/Auth"),
        ("account", "Users/Auth"),
        ("auth", "Users/Auth"),
        ("order", "Orders/Checkout"),
        ("payment", "Payments/Billing"),
        ("invoice", "Payments/Billing"),
        ("product", "Catalog/Products"),
        ("category", "Catalog/Products"),
        ("inventory", "Catalog/Inventory"),
        ("log", "Logging/Audit"),
        ("audit", "Logging/Audit"),
    ]
    table_lookup = {t["name"]: t for t in schema.get("tables", [])}
    summaries = []

    for comp in components:
        labels = defaultdict(int)
        reasons = []
        for tbl in comp:
            low = tbl.lower()
            for kw, label in name_hints:
                if kw in low:
                    labels[label] += 1
            cols = table_lookup.get(tbl, {}).get("columns", [])
            if len(cols) <= 3:
                reasons.append(f"{tbl} is a small table ({len(cols)} cols)")
        label = "Mixed"
        if labels:
            label = max(labels.items(), key=lambda x: x[1])[0]
        if reasons:
            reasons.append(f"Label inferred: {label}")
        else:
            reasons = [f"Label inferred: {label}"]
        summaries.append({"tables": comp, "label": label, "reasons": reasons})
    return summaries

# ---------------------
# Refinement helpers
# ---------------------
def refine_components(components: List[List[str]], schema: Dict[str, Any], usage_map: Dict[tuple, int], usage_threshold: float) -> List[List[str]]:
    relations = schema.get("relations", [])
    refined: List[List[str]] = []
    for comp in components:
        # 1) split large components по сильным usage
        splits = split_component_by_usage(comp, relations, usage_map, usage_threshold)
        for sub in splits:
            # 2) разбить по ключевым словам внутри подкомпонента
            seeded = seed_component_by_keywords(sub, relations)
            refined.extend(seeded)
    # убрать пустые и отсортировать
    refined = [sorted(list(set(c))) for c in refined if c]
    return refined


def split_component_by_usage(component: List[str], relations: List[Dict[str, Any]], usage_map: Dict[tuple, int], usage_threshold: float) -> List[List[str]]:
    """
    Делит большой компонент на подкомпоненты, отсекая слабые связи (usage ниже порога).
    Если после удаления связей остаётся один компонент — возвращает исходный.
    """
    if len(component) <= 5:
        return [sorted(component)]
    component_set = set(component)
    strong_adj = defaultdict(list)
    for r in relations:
        a, b = r.get("from"), r.get("to")
        if a in component_set and b in component_set:
            w = usage_map.get(tuple(sorted((a, b))), 0)
            if w >= usage_threshold:
                strong_adj[a].append(b)
                strong_adj[b].append(a)
    visited = set()
    subcomponents = []
    for node in component:
        if node in visited:
            continue
        stack = [node]
        group = []
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            group.append(n)
            for nb in strong_adj.get(n, []):
                if nb not in visited:
                    stack.append(nb)
        subcomponents.append(sorted(group))
    # если получилось одно или пустые группы — вернуть исходный
    if len(subcomponents) <= 1 or all(len(g) == 0 for g in subcomponents):
        return [sorted(component)]
    # забрать одиночные узлы, которые не имели сильных связей
    unused = component_set - set(sum(subcomponents, []))
    if unused:
        subcomponents.append(sorted(list(unused)))
    return [g for g in subcomponents if g]


def seed_component_by_keywords(component: List[str], relations: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Делит компонент по ключевым словам (users/auth, catalog/products, orders/payments/shipping, promo, reference).
    Затем относит оставшиеся таблицы к ближайшему бакету по связям.
    """
    if len(component) <= 2:
        return [sorted(component)]
    keywords = {
        "users": ("user", "account", "auth", "profile", "session", "role"),
        "catalog": ("product", "category", "inventory", "warehouse", "tag"),
        "orders": ("order", "payment", "shipment", "item", "cart"),
        "promo": ("coupon", "discount", "promo"),
        "reference": ("lookup", "ref", "dict", "status", "audit", "log"),
    }
    buckets = {k: [] for k in keywords}
    remaining = []
    for t in component:
        low = t.lower()
        placed = False
        for bucket, kws in keywords.items():
            if any(k in low for k in kws):
                buckets[bucket].append(t)
                placed = True
                break
        if not placed:
            remaining.append(t)
    non_empty = {k: v for k, v in buckets.items() if v}
    if len(non_empty) <= 1:
        return [sorted(component)]
    # adjacency for attachment
    adj = defaultdict(list)
    comp_set = set(component)
    for r in relations:
        a, b = r.get("from"), r.get("to")
        if a in comp_set and b in comp_set:
            adj[a].append(b)
            adj[b].append(a)
    # attach remaining to the bucket with max adjacency
    for t in remaining:
        scores = []
        for bname, members in non_empty.items():
            score = sum(1 for m in members if m in adj.get(t, []))
            scores.append((score, bname))
        if scores:
            _, best = max(scores, key=lambda x: x[0])
            non_empty[best].append(t)
        else:
            # если нет связей — к самому большому бакету
            largest = max(non_empty.items(), key=lambda x: len(x[1]))[0]
            non_empty[largest].append(t)
    return [sorted(v) for v in non_empty.values() if v]


def attach_small_components(components: List[List[str]], schema: Dict[str, Any], usage_map: Dict[tuple, int], min_size: int = 2) -> List[List[str]]:
    """
    Приклеивает слишком маленькие домены (размер < min_size) к соседям с наибольшим весом связи.
    """
    if not components:
        return components
    rels = schema.get("relations", [])
    # индексация по таблицам
    comp_index = {}
    for idx, comp in enumerate(components):
        for t in comp:
            comp_index[t] = idx
    weights = defaultdict(lambda: defaultdict(int))
    for r in rels:
        a, b = r.get("from"), r.get("to")
        if a in comp_index and b in comp_index:
            ia, ib = comp_index[a], comp_index[b]
            if ia == ib:
                continue
            w = usage_map.get(tuple(sorted((a, b))), 1)
            weights[ia][ib] += w
            weights[ib][ia] += w
    small = []
    large = []
    large_map = {}
    for idx, comp in enumerate(components):
        if len(comp) < min_size:
            small.append((idx, comp))
        else:
            large_map[idx] = len(large)
            large.append(comp)
    if not large:
        return components
    for orig_idx, comp in small:
        if not comp:
            continue
        candidates = weights.get(orig_idx, {})
        target_large_idx = None
        if candidates:
            best = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
            for tgt, _ in best:
                if tgt in large_map:
                    target_large_idx = large_map[tgt]
                    break
        if target_large_idx is None:
            target_large_idx = 0
        large[target_large_idx].extend(comp)
    # финальная очистка
    merged = []
    for comp in large:
        if comp:
            merged.append(sorted(list(set(comp))))
    return merged

# ---------------------
# Metrics calculators
# ---------------------
def compute_metrics(schema: Dict[str, Any], components: List[List[str]]) -> Dict[str, Any]:
    """
    Basic metrics:
    - domain_count
    - avg_internal_coupling: (internal edges)/(internal+external edges) averaged per domain
    - total_cross_domain_edges
    - ownership_conflicts (tables appearing in multiple domains) - should be 0
    - over_segmentation_score: high if many small domains (we use entropy-like measure)
    """
    # build relation set
    rels = [(r["from"], r["to"]) for r in schema.get("relations", [])]
    # map table->comp
    table_to_comp = {}
    for idx, comp in enumerate(components):
        for t in comp:
            table_to_comp[t] = idx

    cross_edges = 0
    comp_internal_counts = defaultdict(int)
    comp_external_counts = defaultdict(int)
    for a,b in rels:
        ia = table_to_comp.get(a, None)
        ib = table_to_comp.get(b, None)
        if ia is None or ib is None:
            continue
        if ia == ib:
            comp_internal_counts[ia] += 1
        else:
            comp_external_counts[ia] += 1
            comp_external_counts[ib] += 1
            cross_edges += 1

    avg_internal_ratio = 0.0
    if components:
        ratios = []
        for idx in range(len(components)):
            internal = comp_internal_counts.get(idx, 0)
            external = comp_external_counts.get(idx, 0)
            denom = internal + external
            ratios.append((internal / denom) if denom > 0 else 1.0)
        avg_internal_ratio = sum(ratios) / len(ratios)

    # over-segmentation: many tiny comps -> high score
    sizes = [len(c) for c in components] if components else [0]
    # entropy-like
    total = sum(sizes) if sizes else 1
    entropy = 0.0
    for s in sizes:
        p = s / total if total > 0 else 0
        if p > 0:
            entropy -= p * math.log(p)
    # normalize entropy by log(n)
    max_entropy = math.log(len(sizes)) if len(sizes) > 1 else 1
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

    ownership_conflicts = 0  # our representation prevents duplicates; keep for future

    metrics = {
        "domain_count": len(components),
        "avg_internal_coupling": round(avg_internal_ratio, 3),
        "cross_domain_edges": cross_edges,
        "over_segmentation_score": round(norm_entropy, 3),
        "ownership_conflicts": ownership_conflicts
    }
    return metrics

# ---------------------
# Comparator
# ---------------------
def compare_results(baseline: Dict[str, Any], ai: Dict[str, Any]) -> Dict[str, Any]:
    b = baseline.get("metrics", {})
    a = ai.get("metrics", {})
    comparison = {}
    keys = set(list(b.keys()) + list(a.keys()))
    for k in keys:
        bv = b.get(k, None)
        av = a.get(k, None)
        # compute delta where numeric
        try:
            delta = None
            if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
                delta = round(av - bv, 3)
            comparison[k] = {"baseline": bv, "ai": av, "delta": delta}
        except Exception:
            comparison[k] = {"baseline": bv, "ai": av, "delta": None}
    # Add simple narrative suggestion
    narrative = []
    if a.get("avg_internal_coupling", 0) > b.get("avg_internal_coupling", 0):
        narrative.append("AI achieves higher internal coupling (better domain cohesion).")
    if a.get("cross_domain_edges", 0) < b.get("cross_domain_edges", 0):
        narrative.append("AI reduces cross-domain FK edges (less coupling).")
    if a.get("domain_count", 0) != b.get("domain_count", 0):
        narrative.append(f"Domain count changed from {b.get('domain_count')} to {a.get('domain_count')}.")
    return {"comparison": comparison, "notes": narrative}

# ---------------------
# Public entrypoint
# ---------------------
def analyze_schema(raw_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and analyze an input schema.
    Returns baseline and heuristic outputs plus comparison.
    """
    schema = deepcopy(raw_schema) if raw_schema else {}
    schema.setdefault("tables", [])
    schema.setdefault("relations", [])
    schema.setdefault("usage", [])

    baseline = baseline_analyze(schema)
    ai = ai_analyze(schema)
    comparison = compare_results(baseline, ai)

    return {
        "baseline": baseline,
        "ai": ai,
        **comparison
    }
