import numpy as np
import networkx as nx
from dataclasses import asdict
import plotly.graph_objects as go
from rich.tree import Tree

__all__ = ["edge_pos",
           "edge_pos_to_array",
           "graph_to_edge_array",
           "edge_array_separated",
           "graph_to_plottable",
           "graph_to_traces",
           "diGraph_to_richTree"]


def edge_pos(edges, pos):
    """ Return a dict of edge endpoints from an iterable of edges and dict of node positions. """
    edge_pos = [(edge, (pos[edge[0]], pos[edge[1]])) for edge in edges]
    return dict(edge_pos)


def edge_pos_to_array(edge_pos):
    """ Return an np array of edge endpoints."""
    a = []
    for edge in edge_pos:
        start, end = edge_pos[edge]
        a.append(start)
        a.append(end)
        # a.append(np.full(2, np.nan))
    return np.array(a)


def graph_to_edge_array(g, layout_func=None):
    """ Return an array of edge endpoints from a graph. """
    if layout_func is None:
        try:
            layout_func = g.layout_func
        except AttributeError:
            layout_func = nx.planar_layout

    return edge_pos_as_array(edge_pos(g.edges(), layout_func(g)))


def edge_array_separated(edge_arr):
    """ Return array with nan between each edge. """
    return np.insert(edge_arr, slice(2, -1, 2), np.full(2, np.nan), axis=0)


def graph_to_plottable(g, layout_func=None):
    if layout_func is None:
        try:
            layout_func = g.layout_func
        except AttributeError:
            if nx.is_planar(g):
                layout_func = nx.planar_layout
            else:
                layout_func = nx.spring_layout

    pos = layout_func(g)
    e_pos_array = edge_array_separated(
        edge_pos_to_array(edge_pos(g.edges(), pos)))
    return (np.array(list(pos.values())),
            e_pos_array)


def graph_to_traces(g, **kwargs):
    try:
        trace_names = kwargs["trace_names"]
        if not g.is_directed() and "arrows" in trace_names:
            raise TypeError("Cannot plot directed arrows if G is undirected."
                            " Unset 'arrows' in 'trace_names'")
    except KeyError:
        trace_names = ["edges", "nodes"]
        if g.is_directed():
            trace_names.append("arrows")

    trace_kwargs = {}
    for name in trace_names:
        # Does nothing but lets us pass below without catching keyerror.
        trace_kwargs[name] = {}
        for kwarg in kwargs:
            if kwarg.startswith(name):
                kwargs = {kwarg.split(name)[1]: kwargs[kwarg]}
                trace_kwargs[name] = kwargs

    try:
        pos, e_pos = graph_to_plottable(g, layout_func=kwargs["layout_func"])
    except KeyError:
        pos, e_pos = graph_to_plottable(g)

    traces = {}
    if "nodes" in trace_names:
        traces["nodes"] = go.Scatter(
            x=pos[:, 0], y=pos[:, 1],
            name="nodes",
            mode="markers",
            **trace_kwargs["nodes"])

    if "edges" in trace_names:
        traces["edges"] = go.Scatter(
            x=e_pos[:, 0], y=e_pos[:, 1],
            name="edges",
            mode="lines",
            **trace_kwargs["edges"])

    return traces


def diGraph_to_richTree(g, branch=None, seen=None, attr=["name"], depth=0):
    """ Take a digraph with parents pointing to children and return a Tree.

    This was really hard to figure out.

    g: The graph to map.
    branch: A reference to the branch of the Tree to add nodes to.
            Usually used internally during recursion.
    seen:   A set of nodes that have alreayd been added to the Tree.
            Prevents duplication of nodes since DiGraphs have references
            to nodes at top level and when they are pointed to.
    attr:   A string identifying the data attribute containing the string to
            add to the tree. Set to `None` to add node. None option
            will fail if str(node) fails.
    depth:  Only usefull for debugging recursion.
    """
    if branch is None:
        branch = Tree(g.graph["name"])
    if seen is None:
        seen = set()

    for n in g:
        if n not in seen:
            # print("\t" * depth, n, g.nodes.data()[n]["name"])
            seen.add(n)
            if attr is None:
                newbranch = Tree(str(g.nodes[n]))
            else:
                for key in attr:
                    try:
                        newbranch = Tree(getattr(g.nodes[n], key))
                    except AttributeError:
                        try:
                            newbranch = Tree(getattr(g.nodes[n]["obj"], key))
                        except AttributeError:
                            pass
                try:
                    newbranch
                except NameError:
                    raise KeyError(
                        f"Key {attr} not in node {n} or in obj attr on node, "
                        "or no obj attr on node.")
            branch.add(diGraph_to_richTree(g.subgraph(
                g.succ[n]), newbranch, seen, attr, depth+1))
    return branch
