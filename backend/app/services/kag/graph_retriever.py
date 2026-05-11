"""Simple graph path retrieval over the NetworkX regulatory graph."""

from __future__ import annotations

from app.services.kag.graph_store import NetworkXGraphStore


class GraphRetriever:
    """Retrieve graph paths by matching query terms to node titles."""

    def __init__(self, store: NetworkXGraphStore):
        self.store = store

    def retrieve_paths(self, query: str, filters: dict | None = None, limit: int = 5) -> list[dict]:
        text = (query or "").lower()
        paths: list[dict] = []

        def _document_regulators(doc_id: str) -> list[str]:
            regulators = []
            for successor in self.store.graph.successors(doc_id):
                attrs = self.store.graph.nodes[successor]
                if attrs.get("node_type") == "Regulator":
                    regulators.append(str(attrs.get("title", successor)))
            return regulators or ["Unknown Regulator"]

        def _append_path(path: list[str], matched_node: str, doc_id: str, topics: list[str]) -> None:
            if len(paths) >= limit:
                return
            paths.append(
                {
                    "path": path,
                    "matched_node": matched_node,
                    "matched_doc_ids": [doc_id] if doc_id else [],
                    "matched_topics": list(dict.fromkeys(topics)),
                }
            )

        for node_id, attributes in self.store.graph.nodes(data=True):
            title = str(attributes.get("title", node_id))
            title_lower = title.lower()
            if title_lower not in text and not any(token in title_lower for token in text.split() if len(token) > 2):
                continue

            node_type = attributes.get("node_type")
            if node_type == "Document":
                doc_topics = [
                    str(self.store.graph.nodes[succ].get("title", succ))
                    for succ in self.store.graph.successors(node_id)
                    if self.store.graph.nodes[succ].get("node_type") == "Topic"
                ]
                for regulator in _document_regulators(node_id):
                    _append_path([regulator, title], node_id, node_id, doc_topics)

            elif node_type in {"Topic", "Product", "Chunk"}:
                for doc_id in self.store.graph.predecessors(node_id):
                    doc_attrs = self.store.graph.nodes[doc_id]
                    if doc_attrs.get("node_type") != "Document":
                        continue
                    doc_title = str(doc_attrs.get("title", doc_id))
                    topics = [title] if node_type == "Topic" else []
                    for regulator in _document_regulators(doc_id):
                        _append_path([regulator, doc_title, title], node_id, doc_id, topics)

            elif node_type == "Regulator":
                for doc_id in self.store.graph.predecessors(node_id):
                    doc_attrs = self.store.graph.nodes[doc_id]
                    if doc_attrs.get("node_type") != "Document":
                        continue
                    doc_title = str(doc_attrs.get("title", doc_id))
                    _append_path([title, doc_title], node_id, doc_id, [])

            if len(paths) >= limit:
                break

        return paths
