import numpy as np
import networkx as nx
from dataclasses import asdict
import plotly.graph_objects as go
from rich.tree import Tree

__all__ = ["edge_pos",
           "edge_pos_to_array",
           "separate_edges",
           "edge_sep_to_mid_data",
           "g_to_edge_array",
           "g_to_plot_arrays",
           "g_to_traces",
           "obj_to_node_and_edges",
           "diGraph_to_richTree",
           "filter_factory"
           ]


def edge_pos(edges, pos):
    """ Return edge endpoints analogous to `pos` from nx.layout funcs."""
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


def separate_edges(edge_arr):
    """ Return array with nan between each edge.

    This makes plotting edges as a single trace in plotly easier.
    """
    return np.insert(edge_arr, slice(2, -1, 2), np.full(2, np.nan), axis=0)


def edge_sep_to_mid_data(g, edge_sep):
    """ Return positions of edge halfway point and edge direction."""
    mid_pos = edge_sep[0::3]/2 + edge_sep[1::3]/2
    mid_vector = edge_sep[1::3] - edge_sep[0::3]
    mid_angle = np.arctan2(mid_vector[:, 0], mid_vector[:, 1]) * (180 / np.pi)
    return (mid_pos, mid_angle)


def g_to_edge_array(g, layout_func=None):
    """ Return an array of edge endpoints from a graph. """
    if layout_func is None:
        try:
            layout_func = g.layout_func
        except AttributeError:
            layout_func = nx.planar_layout

    return edge_pos_to_array(edge_pos(g.edges(), layout_func(g)))


def g_to_plot_arrays(g, layout_func=None):
    if layout_func is None:
        try:
            layout_func = g.layout_func
        except AttributeError:
            if nx.is_planar(g):
                layout_func = nx.planar_layout
            else:
                layout_func = nx.spring_layout

    pos = layout_func(g)
    edge_sep = separate_edges(edge_pos_to_array(edge_pos(g.edges, pos)))
    if g.is_directed():
        mid_data = edge_sep_to_mid_data(g, edge_sep)
    else:
        mid_data = None
    return (np.array(list(pos.values())),
            edge_sep,
            mid_data)


def g_to_traces(g, trace_kwargs={}, layout_func=None):
    trace_defaults = {
        "nodes": {"name": "nodes",
                  "mode": "markers",
                  },
        "edges": {"name": "edges",
                  "mode": "lines",
                  },
        "arrows": {"name": "arrows",
                   "mode": "markers",
                   "marker": {"size": 12,
                              "symbol": "arrow-wide",
                              "color": "green"},
                   }
    }

    # If arrows is explicitly set, let user know that doesn't
    # make sense.
    if "arrows" in trace_kwargs and not g.is_directed():
        raise TypeError("Cannot plot directed arrows if g is undirected. "
                        "Remove `arrows` from trace_kwargs.")

    # This is a little weird, but it doesn't overwrite parameter
    # kwargs with defaults, and keeps the "trace_kwargs".
    kw = trace_defaults
    if not g.is_directed():
        del kw["arrows"]
    kw.update(trace_kwargs)
    trace_kwargs = kw

    # g_to_plot_arrays will handle layout_func if None.
    pos, e_pos, mid_data = g_to_plot_arrays(g, layout_func=layout_func)

    traces = {}
    if "nodes" in trace_kwargs:
        traces["nodes"] = go.Scatter(
            x=pos[:, 0], y=pos[:, 1],
            **trace_kwargs["nodes"])

    if "edges" in trace_kwargs:
        traces["edges"] = go.Scatter(
            x=e_pos[:, 0], y=e_pos[:, 1],
            **trace_kwargs["edges"])

    if "arrows" in trace_kwargs:
        mid_pos, mid_angle = mid_data
        if "angle" not in trace_kwargs["arrows"]["marker"]:
            trace_kwargs["arrows"]["marker"].update({"angle": mid_angle})
        traces["arrows"] = go.Scattergl(
            x=mid_pos[:, 0], y=mid_pos[:, 1],
            **trace_kwargs["arrows"],
        )

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


def diGraph_to_richTree(g, n=None, label_func=None, root_label=None):
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

    if root_label is None:
        try:
            root_label = g.name
        except AttributeError:
            root_label = "Root"

    # Top level might be a forest.
    if n is None:
        nbrs = [node for node in g if len(g.pred[node]) == 0]
        tree = Tree(root_label)
    else:
        nbrs = g.adj[n]
        tree = Tree(label_func(g, n))

    for nb in nbrs:
        if len(g.adj[nb]) > 0:
            tree.add(diGraph_to_richTree(g, nb, label_func))
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
