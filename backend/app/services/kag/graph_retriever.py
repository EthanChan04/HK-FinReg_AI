"""Graph retrieval with entity matching and lightweight multi-hop scoring."""

from __future__ import annotations

import re

import networkx as nx

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
    source_refs: list[dict] = Field(default_factory=list)


class GraphRetriever:
    """Retrieve scored graph paths from the regulatory graph."""

    QUERY_STOPWORDS = {
        "a", "an", "and", "are", "be", "for", "how", "in", "is", "it", "key",
        "of", "on", "the", "to", "what", "which", "who", "with",
    }
    NODE_TYPE_PRIORITY = {
        "RegulatoryTriple": 7,
        "Clause": 6,
        "Section": 5,
        "Topic": 4,
        "RegulatoryDocument": 3,
        "Document": 3,
        "Product": 1,
    }

    def __init__(self, store: NetworkXGraphStore):
        self.store = store

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [
            token.lower()
            for token in re.findall(r"[a-zA-Z0-9_]+", text or "")
            if len(token) > 1 and token.lower() not in GraphRetriever.QUERY_STOPWORDS
        ]

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

    def _candidate_docs_for_node(self, node_id: str, max_hops: int = 4) -> list[str]:
        """Resolve a matched semantic/structure node back to source documents."""

        graph = self.store.graph.to_undirected()
        docs: list[str] = []
        for candidate, distance in nx.single_source_shortest_path_length(graph, node_id, cutoff=max_hops).items():
            if distance == 0:
                continue
            node_type = self.store.graph.nodes[candidate].get("node_type")
            if node_type in {"RegulatoryDocument", "Document"}:
                docs.append(candidate)
        return docs

    def _path_details(self, doc_id: str, node_id: str) -> tuple[list[str], list[str], list[dict]]:
        graph = self.store.graph.to_undirected()
        try:
            node_path = nx.shortest_path(graph, doc_id, node_id)
        except nx.NetworkXNoPath:
            node_path = [doc_id, node_id]
        titles = [str(self.store.graph.nodes[node].get("title", node)) for node in node_path]
        relations: list[str] = []
        refs: list[dict] = []
        for source, target in zip(node_path, node_path[1:]):
            edge = self.store.graph.get_edge_data(source, target)
            if edge is None:
                edge = self.store.graph.get_edge_data(target, source) or {}
            relations.append(str(edge.get("relation", "")))
        for node in node_path:
            attrs = self.store.graph.nodes[node]
            if attrs.get("node_type") in {"Clause", "EvidenceChunk", "RegulatoryTriple"}:
                source = attrs.get("source") or {}
                refs.append(
                    {
                        "doc_id": attrs.get("doc_id") or source.get("doc_id", doc_id),
                        "clause_id": attrs.get("clause_id") or source.get("clause_id"),
                        "page": attrs.get("page") or source.get("page"),
                        "official_url": attrs.get("source_url") or source.get("official_url"),
                    }
                )
        return titles, relations, refs

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

    def retrieve_paths(
        self,
        query: str,
        filters: dict | None = None,
        limit: int = 5,
        include_provenance: bool = True,
    ) -> list[dict]:
        tokens = self._tokens(query)
        results: list[GraphPathResult] = []
        seen: set[tuple[str, str]] = set()

        matched_nodes: list[tuple[float, str, dict]] = []
        for node_id, attributes in self.store.graph.nodes(data=True):
            title = str(attributes.get("title", node_id))
            node_match_score = self._node_match_score(tokens, title)
            if node_match_score <= 0:
                continue
            matched_nodes.append((node_match_score, node_id, attributes))

        # A broad regulatory corpus contains hundreds of chunks with generic
        # terms such as "requirement". Bound graph expansion after ranking so
        # each query has predictable latency and cannot trigger path explosion.
        candidate_budget = max(limit * 3, 12)
        for node_match_score, node_id, attributes in sorted(
            matched_nodes,
            key=lambda item: (
                item[0],
                self.NODE_TYPE_PRIORITY.get(item[2].get("node_type", ""), 0),
            ),
            reverse=True,
        )[:candidate_budget]:

            node_type = attributes.get("node_type")
            title = str(attributes.get("title", node_id))

            candidate_docs: list[str] = []
            if node_type in {"RegulatoryDocument", "Document"}:
                candidate_docs.append(node_id)
            else:
                candidate_docs.extend(self._candidate_docs_for_node(node_id))

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

                if include_provenance:
                    titles, path_relations, source_refs = self._path_details(doc_id, node_id)
                else:
                    titles = [doc_title, title]
                    path_relations = []
                    source_refs = []
                tail = [] if title == doc_title else [title]
                path = regulator_titles[:1] + ([doc_title] if titles else []) + tail
                relation_chain += path_relations + neighbors["relations"][:2]
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
                        source_refs=source_refs,
                        explanation=(
                            f"Matched node '{title}' with document '{doc_title}' "
                            f"and {len(neighbors['obligations'])} obligations."
                        ),
                    )
                )

        ranked = sorted(results, key=lambda item: item.confidence, reverse=True)[:limit]
        return [item.model_dump() for item in ranked]
