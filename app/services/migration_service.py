def generate_migration_plan(analysis_results: dict):
    """
    Takes analysis output and returns:
    - service boundaries
    - ownership decisions
    - suggested DB splits
    """
    analysis = analysis_results.get("analysis") if isinstance(analysis_results, dict) else None
    analysis = analysis or analysis_results or {}

    ai = analysis.get("ai", {})
    domains = ai.get("domains", [])
    domain_summaries = ai.get("domain_summaries", [])
    metrics = ai.get("metrics", {})

    services = []
    ownership_map = {}
    for idx, domain in enumerate(domains):
        service_name = f"service_{idx+1}"
        summary = domain_summaries[idx] if idx < len(domain_summaries) else {}
        services.append({
            "name": service_name,
            "tables": domain,
            "label": summary.get("label"),
            "reasons": summary.get("reasons", [])
        })
        for table in domain:
            ownership_map[table] = service_name

    db_split_plan = {
        "shards": len(domains),
        "mapping": ownership_map
    }

    notes = [
        f"Total services: {len(domains)}",
        f"Avg internal coupling: {metrics.get('avg_internal_coupling')}",
        f"Cross-domain edges: {metrics.get('cross_domain_edges')}"
    ]

    return {
        "services": services,
        "ownership_map": ownership_map,
        "db_split_plan": db_split_plan,
        "notes": "; ".join(notes)
    }
