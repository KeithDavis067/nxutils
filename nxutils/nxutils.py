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
           "diGraph_to_richTree",
           "obj_to_node_and_edges"]


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


def edge_sep_to_mid_data(g, edge_sep):
    mid_pos = edge_sep[0::3]/2 + edge_sep[1::3]/2
    mid_vector = edge_sep[1::3] - edge_sep[0::3]
    mid_angle = np.arctan2(mid_vector[:, 0], mid_vector[:, 1]) * (180 / np.pi)
    return (mid_pos, mid_angle)


def graph_to_edge_array(g, layout_func=None):
    """ Return an array of edge endpoints from a graph. """
    if layout_func is None:
        try:
            layout_func = g.layout_func
        except AttributeError:
            layout_func = nx.planar_layout

    return edge_pos_to_array(edge_pos(g.edges(), layout_func(g)))


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
    edge_sep = edge_array_separated(edge_pos_to_array(edge_pos(g.edges, pos)))
    if g.is_directed():
        mid_data = edge_sep_to_mid_data(g, edge_sep)
    else:
        mid_data = None
    return (np.array(list(pos.values())),
            edge_sep,
            mid_data)


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
        pos, e_pos, mid_data = graph_to_plottable(
            g, layout_func=kwargs["layout_func"])
    except KeyError:
        pos, e_pos, mid_data = graph_to_plottable(g)

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

    if "arrows" in trace_names:
        mid_pos, mid_angle = mid_data
        traces["arrows"] = go.Scattergl(
            x=mid_pos[:, 0], y=mid_pos[:, 1],
            mode="markers",
            marker=dict(size=12,
                        symbol="arrow-wide",
                        angle=mid_angle,
                        color="green",
                        ),
            name="arrows")

    return traces


def obj_to_node_and_edges(obj, node_attr,
                          parent_attr_list=["parent_id",
                                            "project_id",
                                            "section_id"],
                          edge_attr=None,
                          edge_attr_func=None):
    """ Return a node and edges suitable for adding to a graph.

    tdobj: an todoist object, or an object with an `id` attribute
        suitable as a nx.DiGraph node
        holding a node id to point to.
    node_attr: attribute to use as node. Set explicitly to `None` to use
        obj as node. Nodes must follow `nx.Graph` rules such as hashability.
    parent_attr_list: list of object attributes to use as edge endpoints.
        If none of the listed attributes exist, no edge is added.
    edge_attr: A list or dict of attributes on `obj` to assign as data on the edge.
        Set edge_attr to `False` (must be the False object) to ignore edge_attr_func.
    edge_attr_func: A function that accepts only the object and returns a dict
        of edge attributes names and values. Will `update` any dict created by `edge_attr`
        and therefore override any values of the same key.

    Returns:
        A tuple of the id and data suitable for adding to a graph, and a
        dict mapping `parent_attr_list` to the edge data. The function adding
        the returned to a DiGraph should create a list of edges from the edge dict.
    """
    # If an attribute  on `obj` is to be used as the node, extract it.
    if node_attr is None:
        node = obj
    else:
        node = getattr(obj, node_attr)
    # Each `parent_attr` may add an edge.
    edgebunch = []
    for parent_attr in parent_attr_list:
        try:
            parent = getattr(obj, parent_attr)
        except AttributeError:
            continue  # Skip if attribute doesn't exist.

        if parent is not None:  # Skip: [u][None] isn't an edge.
            edge = (node, parent)
            if edge_attr is not False:  # Add no edge data if edge_attr False.
                if edge_attr is not None:
                    # Allow list or dict.
                    try:
                        eattr_items = edge_attr.items()
                    except AttributeError:
                        eattr_items = zip(edge_attr, edge_attr)
                    for eattr_name, eattr in eattr_items:
                        eattr = getattr(obj, eattr)
                        # Don't set None, but don't raise error.
                        if eattr is not None:
                            edge = (*edge, {eattr_name: getattr(obj, eattr)})

                if edge_attr_func is not None:
                    try:
                        u, v, data = edge
                    except ValueError:
                        u, v = edge
                    # edge_attr_func must return a dict.
                    try:
                        data.update(edge_attr_func(obj))
                    except (NameError, UnboundLocalError):
                        data = edge_attr_func(obj)

                    edge = (u, v, data)
        try:
            edgebunch.append(edge)
        except NameError:
            pass

    return (node, dict(obj=obj)), edgebunch


def diGraph_to_richTree(g, n=None, label_func=None):
    """ Take a digraph with parents pointing to children and return a Tree.

    This was really hard to figure out.
    Recursively walk a graph and return a Rich.Tree
    of the hierarchy.

    g: The graph to walk.
    n: The first node in the walk. If `None` self to the first node
        returned by `list(g.nodes)[0].`
    label_func: A function that takes a graph and a node and returns
        a string used as the label for that node in the `Rich.Tree`.
        If `None`, try to convert node to a string and use as the label.

    Returns:
        a `Rich.Tree`
    """

    if label_func is None:
        def label_func(g, n):
            return str(n)

    if n is None:
        n = list(g.nodes)[0]

    tree = Tree(label_func(g, n))

    for nb in g.adj[n]:
        if len(g.adj[nb]) > 0:
            tree.add(diGraph_to_richTree(g, nb))
        else:
            tree.add(label_func(g, nb))
    return tree


def filter_factory(G, attr, value):
    def filter(node):
        try:
            if G.nodes[node][attr] == value:
                return True
            else:
                return False
        except KeyError:
            try:
                if getattr(G.nodes[node], attr) == value:
                    return True
                else:
                    return False
            except AttributeError:
                return False
        return False
    return filter
