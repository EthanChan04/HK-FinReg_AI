"""Multi-hop path reasoning over the NetworkX regulatory graph."""

from __future__ import annotations

from app.services.kag.graph_store import NetworkXGraphStore


def find_reasoning_paths(
    seed_node_ids: list[str],
    store: NetworkXGraphStore,
    max_hops: int = 2,
) -> list[dict]:
    """Find multi-hop reasoning paths from seed nodes through the graph.

    For each seed node, walks up to *max_hops* hops via predecessors and
    successors, collecting structured paths with source, target and relation
    metadata.
    """
    if not seed_node_ids or store.graph is None:
        return []

    graph = store.graph
    seen = set()
    paths: list[dict] = []

    for seed_id in seed_node_ids:
        if seed_id not in graph:
            continue
        seed_attrs = graph.nodes.get(seed_id, {})
        seed_title = seed_attrs.get("title", seed_id)

        # Walk predecessors (documents / regulators that point to this node)
        for pred in graph.predecessors(seed_id):
            if (pred, seed_id) in seen:
                continue
            seen.add((pred, seed_id))
            pred_attrs = graph.nodes.get(pred, {})
            edge_data = graph.get_edge_data(pred, seed_id) or {}
            paths.append({
                "source": pred,
                "source_type": pred_attrs.get("node_type", "unknown"),
                "source_title": pred_attrs.get("title", pred),
                "target": seed_id,
                "target_type": seed_attrs.get("node_type", "unknown"),
                "target_title": seed_title,
                "relation": edge_data.get("relation", "unknown"),
                "hops": 1,
            })

            # Walk one more hop from predecessor in both directions
            if max_hops >= 2:
                # Backward: predecessors of the predecessor
                for pred2 in graph.predecessors(pred):
                    if (pred2, pred) in seen:
                        continue
                    seen.add((pred2, pred))
                    pred2_attrs = graph.nodes.get(pred2, {})
                    edge_data2 = graph.get_edge_data(pred2, pred) or {}
                    paths.append({
                        "source": pred2,
                        "source_type": pred2_attrs.get("node_type", "unknown"),
                        "source_title": pred2_attrs.get("title", pred2),
                        "target": pred,
                        "target_type": pred_attrs.get("node_type", "unknown"),
                        "target_title": pred_attrs.get("title", pred),
                        "relation": edge_data2.get("relation", "unknown"),
                        "hops": 2,
                    })
                # Forward: successors of the predecessor (e.g., regulator from doc)
                for succ2 in graph.successors(pred):
                    if succ2 == seed_id:
                        continue  # already captured as 1-hop
                    if (pred, succ2) in seen:
                        continue
                    seen.add((pred, succ2))
                    succ2_attrs = graph.nodes.get(succ2, {})
                    edge_data2 = graph.get_edge_data(pred, succ2) or {}
                    paths.append({
                        "source": pred,
                        "source_type": pred_attrs.get("node_type", "unknown"),
                        "source_title": pred_attrs.get("title", pred),
                        "target": succ2,
                        "target_type": succ2_attrs.get("node_type", "unknown"),
                        "target_title": succ2_attrs.get("title", succ2),
                        "relation": edge_data2.get("relation", "unknown"),
                        "hops": 2,
                    })

        # Walk successors (topics / obligations that the seed node points to)
        for succ in graph.successors(seed_id):
            if (seed_id, succ) in seen:
                continue
            seen.add((seed_id, succ))
            succ_attrs = graph.nodes.get(succ, {})
            edge_data = graph.get_edge_data(seed_id, succ) or {}
            paths.append({
                "source": seed_id,
                "source_type": seed_attrs.get("node_type", "unknown"),
                "source_title": seed_title,
                "target": succ,
                "target_type": succ_attrs.get("node_type", "unknown"),
                "target_title": succ_attrs.get("title", succ),
                "relation": edge_data.get("relation", "unknown"),
                "hops": 1,
            })

    return paths
