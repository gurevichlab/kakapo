"""Graph-theoretic algorithms for molecular fragmentation.

This module implements the optimal O(N+M) + O(M^2) graph-theoretic algorithm using
random XOR-hashing to identify all bridges (1-cuts) and 2-cuts in an undirected,
connected graph.
"""

import random
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx

from kakapo.utils.logger import get_logger

logger = get_logger(__name__)

Edge = tuple[int, int]
EdgePair = tuple[Edge, Edge]


@dataclass(frozen=True)
class BridgeCut:
    """Represents a bridge (1-cut) with its partition masks."""

    edge: Edge
    mask1: int
    mask2: int


@dataclass(frozen=True)
class TwoCut:
    """Represents a 2-cut with its partition masks."""

    edge_pair: EdgePair
    mask1: int
    mask2: int


@dataclass(frozen=True)
class CutSearchResult:
    """Container for the results of the cut search algorithm."""

    bridges: list[BridgeCut]
    two_cuts: list[TwoCut]


def _simplify_graph(graph: nx.Graph) -> nx.Graph:
    """Simplify the input graph: make undirected, remove self-loops and multi-edges."""
    if graph.is_directed():
        undirected_graph = graph.to_undirected()
    else:
        undirected_graph = graph

    if isinstance(undirected_graph, nx.MultiGraph):
        simple_graph = nx.Graph(undirected_graph)
    else:
        simple_graph = undirected_graph

    self_loops = list(nx.selfloop_edges(simple_graph))
    if self_loops:
        simple_graph.remove_edges_from(self_loops)
    return simple_graph


def _run_dfs(
    graph: nx.Graph,
) -> tuple[dict[int, int], dict[int, int | None], dict[int, list[int]], list[Edge]]:
    """Perform DFS traversal to identify tree structure and back-edges."""
    visited = set()
    tin = {}
    timer = 0
    parent = {}
    children = defaultdict(list)
    back_edges = []

    def dfs(u: int, p: int | None = None) -> None:
        nonlocal timer
        visited.add(u)
        tin[u] = timer
        timer += 1
        parent[u] = p
        for v in graph.neighbors(u):
            if v == p:
                continue
            if v not in visited:
                children[u].append(v)
                dfs(v, u)
            elif tin[v] < tin[u]:
                back_edges.append((u, v))

    if len(graph.nodes) > 0:
        start_node = next(iter(graph.nodes))
        dfs(start_node)

    return tin, parent, children, back_edges


def _compute_accumulators(
    nodes: nx.classes.reportviews.NodeView, back_edges: list[Edge]
) -> tuple[dict[int, int], dict[Edge, int]]:
    """Assign random 64-bit weights to back-edges and compute vertex accumulators."""
    back_edge_weights = {}
    vertex_accumulators = dict.fromkeys(nodes, 0)
    for u, v in back_edges:
        canonical = (u, v) if u < v else (v, u)
        w = 0
        while w == 0:
            w = random.getrandbits(64)
        back_edge_weights[canonical] = w
        vertex_accumulators[u] ^= w
        vertex_accumulators[v] ^= w
    return vertex_accumulators, back_edge_weights


def _subtree_aggregation(
    nodes: nx.classes.reportviews.NodeView,
    tin: dict[int, int],
    parent: dict[int, int | None],
    children: dict[int, list[int]],
    vertex_accumulators: dict[int, int],
    back_edge_weights: dict[Edge, int],
) -> dict[Edge, int]:
    """Perform bottom-up aggregation to compute final edge weights."""
    nodes_post_order = sorted(nodes, key=lambda node: tin[node], reverse=True)
    subtree_xor = {}
    all_edge_weights = {}

    for u in nodes_post_order:
        val = vertex_accumulators[u]
        for child in children[u]:
            val ^= subtree_xor[child]
        subtree_xor[u] = val

        p = parent[u]
        if p is not None:
            edge = (p, u) if p < u else (u, p)
            all_edge_weights[edge] = val

    for edge, w in back_edge_weights.items():
        all_edge_weights[edge] = w

    return all_edge_weights


def _get_2cut_mask(
    edge1: Edge,
    edge2: Edge,
    tree_edges_set: set[Edge],
    tin: dict[int, int],
    subtree_masks: dict[int, int],
) -> int | None:
    """Compute the partition mask for a 2-cut pair."""
    is1_tree = edge1 in tree_edges_set
    is2_tree = edge2 in tree_edges_set

    if is1_tree and not is2_tree:
        u, v = edge1
        child = u if tin[u] > tin[v] else v
        return subtree_masks[child]

    if is2_tree and not is1_tree:
        u, v = edge2
        child = u if tin[u] > tin[v] else v
        return subtree_masks[child]

    if is1_tree and is2_tree:
        u1, v1 = edge1
        child1 = u1 if tin[u1] > tin[v1] else v1
        u2, v2 = edge2
        child2 = u2 if tin[u2] > tin[v2] else v2

        if tin[child1] < tin[child2]:
            return subtree_masks[child1] ^ subtree_masks[child2]
        return subtree_masks[child2] ^ subtree_masks[child1]

    return None


def _classify_cuts(
    all_edge_weights: dict[Edge, int],
    tin: dict[int, int],
    subtree_masks: dict[int, int],
    total_mask: int,
    tree_edges_set: set[Edge],
) -> tuple[list[BridgeCut], list[TwoCut]]:
    """Classify cuts into BridgeCut and TwoCut objects with O(1) component splits."""
    groups = defaultdict(list)
    bridges_list = []

    for edge, w in all_edge_weights.items():
        if w == 0:
            # For tree edge (parent, child), child is the one with larger tin
            u, v = edge
            child = u if tin[u] > tin[v] else v
            mask1 = subtree_masks[child]
            mask2 = total_mask ^ mask1
            bridges_list.append(BridgeCut(edge=edge, mask1=mask1, mask2=mask2))
        else:
            groups[w].append(edge)

    two_cuts_list = []
    for _, edges in groups.items():
        if len(edges) >= 2:
            m = len(edges)
            for i in range(m):
                for j in range(i + 1, m):
                    edge1 = edges[i]
                    edge2 = edges[j]
                    edge_pair = (edge1, edge2) if edge1 < edge2 else (edge2, edge1)

                    mask1 = _get_2cut_mask(edge1, edge2, tree_edges_set, tin, subtree_masks)
                    if mask1 is None:
                        continue

                    mask2 = total_mask ^ mask1
                    two_cuts_list.append(TwoCut(edge_pair=edge_pair, mask1=mask1, mask2=mask2))

    return bridges_list, two_cuts_list


def find_bridges_and_2cuts(
    graph: nx.Graph,
) -> CutSearchResult:
    """Find all bridges and 2-cuts in an undirected connected graph with masks.

    Uses a random XOR-hashing approach:
    1. Perform DFS to generate a spanning tree and identify back-edges.
    2. Assign a random 64-bit weight to each back-edge. Tree edges start with 0.
    3. Propagate weights bottom-up: the weight of a tree edge (parent, u) is the
       XOR sum of accumulators of all nodes in u's subtree.
    4. Group edges by their XOR weight:
       - Weight 0: Bridges.
       - Non-zero weight shared by multiple edges: Minimal 2-cuts.

    Args:
        graph: An undirected NetworkX graph representing the molecule.
            We assume it has no self-loops and is connected.

    Returns:
        A CutSearchResult containing lists of BridgeCut and TwoCut.
    """
    simple_graph = _simplify_graph(graph)
    num_nodes = len(simple_graph.nodes)
    if num_nodes == 0:
        return CutSearchResult([], [])

    total_mask = (1 << num_nodes) - 1

    if num_nodes == 1:
        return CutSearchResult([], [])

    # Spanning Tree Generation via DFS
    tin, parent, children, back_edges = _run_dfs(simple_graph)

    # Precompute tree edges set for quick lookup
    tree_edges_set = set()
    for child, p in parent.items():
        if p is not None:
            edge = (p, child) if p < child else (child, p)
            tree_edges_set.add(edge)

    # Bottom-Up Bitmask Aggregation
    nodes_post_order = sorted(simple_graph.nodes, key=lambda node: tin[node], reverse=True)
    subtree_masks = {}
    for u in nodes_post_order:
        mask = 1 << u
        for child in children[u]:
            mask |= subtree_masks[child]
        subtree_masks[u] = mask

    # XOR-hashing Cut Detection
    vertex_accumulators, back_edge_weights = _compute_accumulators(simple_graph.nodes, back_edges)
    all_edge_weights = _subtree_aggregation(
        simple_graph.nodes, tin, parent, children, vertex_accumulators, back_edge_weights
    )

    # Classify cuts and calculate splits
    bridges, two_cuts = _classify_cuts(
        all_edge_weights, tin, subtree_masks, total_mask, tree_edges_set
    )

    logger.debug(
        "XOR-hashing: Found %d bridge(s) and %d 2-cut pair(s) in graph of size V=%d, E=%d",
        len(bridges),
        len(two_cuts),
        num_nodes,
        len(simple_graph.edges),
    )

    return CutSearchResult(bridges=bridges, two_cuts=two_cuts)
