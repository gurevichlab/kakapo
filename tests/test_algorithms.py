"""Unit tests for graph-theoretic algorithms.

Tests the optimal O(N+M) XOR-hashing algorithm for finding bridges and 2-cuts.
"""

import networkx as nx

from kakapo.core.algorithms import find_bridges_and_2cuts


def test_empty_or_single_node():
    """Graph with 0 or 1 nodes should have no bridges or 2-cuts."""
    g_empty = nx.Graph()
    result = find_bridges_and_2cuts(g_empty)
    assert result.bridges == []
    assert result.two_cuts == []

    g_single = nx.Graph()
    g_single.add_node(1)
    result = find_bridges_and_2cuts(g_single)
    assert result.bridges == []
    assert result.two_cuts == []


def test_linear_path():
    """A linear path graph has only bridges and no 2-cuts."""
    g = nx.path_graph(4)  # 0 - 1 - 2 - 3
    result = find_bridges_and_2cuts(g)
    bridges = result.bridges
    two_cuts = result.two_cuts

    # All edges should be bridges
    expected_edges = {(0, 1), (1, 2), (2, 3)}
    assert {b.edge for b in bridges} == expected_edges
    assert two_cuts == []

    # Verify that each bridge splits the graph into two valid components
    total_mask = (1 << len(g.nodes)) - 1
    for bridge in bridges:
        assert bridge.mask1 | bridge.mask2 == total_mask
        assert bridge.mask1 & bridge.mask2 == 0


def test_simple_cycle():
    """A simple 3-cycle should have no bridges, and 3 2-cuts."""
    g = nx.cycle_graph(3)  # 0 - 1 - 2 - 0
    result = find_bridges_and_2cuts(g)
    bridges = result.bridges
    two_cuts = result.two_cuts

    assert bridges == []

    # Any pair of edges from the 3-cycle is a 2-cut
    e1 = (0, 1)
    e2 = (0, 2)
    e3 = (1, 2)
    expected_two_cuts = {
        (e1, e2),
        (e1, e3),
        (e2, e3),
    }

    # Verify each 2-cut is canonicalized and has valid masks
    total_mask = (1 << len(g.nodes)) - 1
    for cut in two_cuts:
        assert cut.edge_pair[0] < cut.edge_pair[1]
        assert cut.edge_pair[0][0] < cut.edge_pair[0][1]
        assert cut.edge_pair[1][0] < cut.edge_pair[1][1]
        assert cut.mask1 | cut.mask2 == total_mask
        assert cut.mask1 & cut.mask2 == 0

    assert {cut.edge_pair for cut in two_cuts} == expected_two_cuts


def test_mixed_structure():
    """A bridge connecting a node to a cycle: 0 - 1 - 2 - 3 - 1."""
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 1)])
    result = find_bridges_and_2cuts(g)
    bridges = result.bridges
    two_cuts = result.two_cuts

    # (0, 1) is a bridge
    assert {b.edge for b in bridges} == {(0, 1)}

    # Cycle 1-2-3-1 has edges (1, 2), (2, 3), (1, 3) canonicalized
    e1 = (1, 2)
    e2 = (1, 3)
    e3 = (2, 3)
    expected_two_cuts = {
        (e1, e2),
        (e1, e3),
        (e2, e3),
    }
    assert {cut.edge_pair for cut in two_cuts} == expected_two_cuts


def test_multi_edges():
    """Parallel edges between same vertices should be treated as a single edge."""
    g = nx.MultiGraph()
    g.add_edge(0, 1)
    g.add_edge(0, 1)  # parallel edge
    g.add_edge(1, 2)

    result = find_bridges_and_2cuts(g)
    bridges = result.bridges
    two_cuts = result.two_cuts

    # The simplified graph is 0 - 1 - 2, which has bridges (0, 1) and (1, 2)
    assert {b.edge for b in bridges} == {(0, 1), (1, 2)}
    assert two_cuts == []


def test_theta_graph():
    """Theta graph (two cycles sharing a path).

    E.g. nodes 0, 1, 2, 3. Edges:
    - Path 1: 0 - 1 - 2
    - Path 2: 0 - 2 (direct)
    - Path 3: 0 - 3 - 2
    This forms cycles (0-1-2-0), (0-2-3-0), and (0-1-2-3-0).
    """
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (0, 2), (0, 3), (2, 3)])
    result = find_bridges_and_2cuts(g)
    bridges = result.bridges
    two_cuts = result.two_cuts

    # No bridges because removing any single edge leaves the graph connected.
    assert bridges == []

    # Valid 2-cuts are:
    # 1. Removing two edges from the same path: e.g. {(0, 1), (1, 2)}
    # 2. Removing one edge from each cycle if they cut the remaining path, e.g.
    #    - (0, 3) and (2, 3) isolates node 3
    #    - (0, 1) and (1, 2) isolates node 1
    # Verify that the 2-cuts found partition the graph correctly.
    for cut in two_cuts:
        g_copy = g.copy()
        g_copy.remove_edge(*cut.edge_pair[0])
        g_copy.remove_edge(*cut.edge_pair[1])
        assert not nx.is_connected(g_copy)

        # Check mask consistency
        total_mask = (1 << len(g.nodes)) - 1
        assert cut.mask1 | cut.mask2 == total_mask
        assert cut.mask1 & cut.mask2 == 0
