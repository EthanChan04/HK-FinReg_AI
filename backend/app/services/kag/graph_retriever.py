"""Graph retrieval with entity matching and lightweight multi-hop scoring."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.services.kag.graph_store import NetworkXGraphStore
from app.services.retrieval.retrieval_service import priority_boost


class GraphPathResult(BaseModel):
    path: list[str]
    relation_chain: list[str] = Field(default_factory=list)
    matched_node: str
    matched_doc_ids: list[str] = Field(default_factory=list)
    matched_topics: list[str] = Field(default_factory=list)
    matched_obligations: list[str] = Field(default_factory=list)
    matched_risks: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""


class GraphRetriever:
    """Retrieve scored graph paths from the regulatory graph."""

    def __init__(self, store: NetworkXGraphStore):
        self.store = store

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.lower() for token in re.findall(r"[a-zA-Z0-9_]+", text or "") if len(token) > 1]

    def _node_match_score(self, query_tokens: list[str], title: str) -> float:
        title_lower = title.lower()
        if not query_tokens:
            return 0.0
        hits = sum(1 for token in query_tokens if token in title_lower)
        return hits / len(query_tokens)

    def _doc_neighbors(self, doc_id: str) -> dict[str, list[str]]:
        topics: list[str] = []
        obligations: list[str] = []
        risks: list[str] = []
        relations: list[str] = []
        for succ in self.store.graph.successors(doc_id):
            attrs = self.store.graph.nodes[succ]
            edge = self.store.graph.get_edge_data(doc_id, succ) or {}
            relation = str(edge.get("relation", ""))
            node_type = attrs.get("node_type")
            title = str(attrs.get("title", succ))
            relations.append(relation)
            if node_type in {"Topic"}:
                topics.append(title)
            if node_type in {"Obligation"}:
                obligations.append(title)
            if node_type in {"Risk"}:
                risks.append(title)
        return {"topics": topics, "obligations": obligations, "risks": risks, "relations": relations}

    @staticmethod
    def _metadata_matches(doc_attrs: dict, filters: dict | None) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            if expected in (None, [], ""):
                continue
            expected_values = expected if isinstance(expected, list) else [expected]
            actual = doc_attrs.get(key)
            if isinstance(actual, list):
                actual_values = actual
            elif isinstance(actual, str) and "," in actual:
                actual_values = [value.strip() for value in actual.split(",")]
            else:
                actual_values = [actual]
            actual_normalized = {str(v).lower() for v in actual_values if v is not None}
            expected_normalized = {str(v).lower() for v in expected_values if v is not None}
            if expected_normalized and not actual_normalized.intersection(expected_normalized):
                return False
        return True

    def retrieve_paths(self, query: str, filters: dict | None = None, limit: int = 5) -> list[dict]:
        tokens = self._tokens(query)
        results: list[GraphPathResult] = []
        seen: set[tuple[str, str]] = set()

        for node_id, attributes in self.store.graph.nodes(data=True):
            title = str(attributes.get("title", node_id))
            node_match_score = self._node_match_score(tokens, title)
            if node_match_score <= 0:
                continue

            node_type = attributes.get("node_type")

            candidate_docs: list[str] = []
            if node_type in {"RegulatoryDocument", "Document"}:
                candidate_docs.append(node_id)
            else:
                for pred in self.store.graph.predecessors(node_id):
                    pred_attrs = self.store.graph.nodes[pred]
                    if pred_attrs.get("node_type") in {"RegulatoryDocument", "Document"}:
                        candidate_docs.append(pred)

            for doc_id in candidate_docs:
                doc_attrs = self.store.graph.nodes[doc_id]
                if not self._metadata_matches(doc_attrs, filters):
                    continue
                doc_title = str(doc_attrs.get("title", doc_id))
                neighbors = self._doc_neighbors(doc_id)

                regulator_titles: list[str] = []
                relation_chain = []
                for succ in self.store.graph.successors(doc_id):
                    succ_attrs = self.store.graph.nodes[succ]
                    edge = self.store.graph.get_edge_data(doc_id, succ) or {}
                    if succ_attrs.get("node_type") == "Regulator":
                        regulator_titles.append(str(succ_attrs.get("title", succ)))
                        relation_chain.append(str(edge.get("relation", "")))

                regulator_match_score = 1.0 if any(
                    self._node_match_score(tokens, regulator) > 0 for regulator in regulator_titles
                ) else 0.0
                obligation_match_score = 1.0 if any(
                    self._node_match_score(tokens, value) > 0 for value in neighbors["obligations"]
                ) else 0.0
                evidence_support_score = 1.0 if any(
                    relation == "SUPPORTED_BY" for relation in neighbors["relations"]
                ) else 0.0

                confidence = (
                    0.35 * node_match_score
                    + 0.25 * regulator_match_score
                    + 0.20 * obligation_match_score
                    + 0.10 * evidence_support_score
                    + 0.10 * priority_boost(doc_attrs)
                )

                tail = [] if title == doc_title else [title]
                path = regulator_titles[:1] + [doc_title] + tail
                relation_chain += neighbors["relations"][:2]
                dedupe_key = (doc_id, title)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                results.append(
                    GraphPathResult(
                        path=path,
                        relation_chain=relation_chain,
                        matched_node=node_id,
                        matched_doc_ids=[doc_id],
                        matched_topics=neighbors["topics"],
                        matched_obligations=neighbors["obligations"],
                        matched_risks=neighbors["risks"],
                        confidence=round(confidence, 3),
                        explanation=(
                            f"Matched node '{title}' with document '{doc_title}' "
                            f"and {len(neighbors['obligations'])} obligations."
                        ),
                    )
                )

        ranked = sorted(results, key=lambda item: item.confidence, reverse=True)[:limit]
        return [item.model_dump() for item in ranked]
