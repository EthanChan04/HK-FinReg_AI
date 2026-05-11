"""NetworkX graph store with node-link JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph


class NetworkXGraphStore:
    """Small graph store suitable for local KAG prototyping."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.graph = nx.DiGraph()

    def add_node(self, node_id: str, **attributes) -> None:
        self.graph.add_node(node_id, **attributes)

    def add_edge(self, source: str, target: str, relation: str, **attributes) -> None:
        self.graph.add_edge(source, target, relation=relation, **attributes)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json_graph.node_link_data(self.graph, edges="links")
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.graph = json_graph.node_link_graph(payload, edges="links", directed=True)
