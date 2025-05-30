from hashlib import blake2b

import torch
from torch_geometric.utils import scatter



def random_walk_pe_with_node_types(data, walk_length: int):
    '''
    Random walk positional encoding that considers node types as described in
    Ma et al., Baking Symmetry into GFlowNets, page 4.
    This code is adapted from `torch_geometric.transforms.add_positional_encoding`.
    '''
    assert data.edge_index is not None
    row, col = data.edge_index
    N = data.num_nodes
    assert N is not None

    color = torch.log(data.node_types.float() + 2.0)
    if len(color) == 0:
        return color

    if data.edge_weight is None:
        value = torch.ones(data.num_edges, device=row.device)
    else:
        value = data.edge_weight
    value = scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]
    value = 1.0 / value

    adj = torch.zeros((N, N), device=row.device)
    adj[row, col] = value
    
    out = adj
    pe_list = [color]
    for _ in range(walk_length - 1):
        out = out @ adj
        colored_rwpe = (color @ out)
        pe_list.append(colored_rwpe)

    pe = torch.stack(pe_list, dim=-1)
    return pe


def full_random_walk_pe(data, walk_length: int):
    '''
    Random walk positional encodings, including off-diagonal elements.
    This can be used to enhance edge-level features.
    This code is adapted from `torch_geometric.transforms.add_positional_encoding`.
    '''
    assert data.edge_index is not None
    row, col = data.edge_index
    N = data.num_nodes
    assert N is not None

    if data.edge_weight is None:
        value = torch.ones(data.num_edges, device=row.device)
    else:
        value = data.edge_weight
    value = scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]
    value = 1.0 / value

    adj = torch.zeros((N, N), device=row.device)
    adj[row, col] = value
    
    out = adj
    pe_list = []
    for _ in range(walk_length):
        out = out @ adj
        pe_list.append(out)
    pe = torch.stack(pe_list, dim=-1)
    return pe


def _hash_label(label, digest_size):
    return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()


def _init_node_labels(G, node_attrs):
    node_labels = {u: str(deg) for u, deg in G.degree()}
    for node_attr in node_attrs:
        node_labels = { 
            u: node_labels[u] + "," + str(dd[node_attr]) for u, dd in G.nodes(data=True)
        }   
    return node_labels


def _neighborhood_aggregate(G, node, node_labels, edge_attr=None):
    """ 
    Compute new labels for given node by aggregating
    the labels of each node's neighbors.
    """
    label_list = []
    for nbr in G.neighbors(node):
        prefix = "" if edge_attr is None else str(G[node][nbr][edge_attr])
        label_list.append(prefix + node_labels[nbr])
    return node_labels[node] + "".join(sorted(label_list))


def weisfeiler_lehman_graph_hash(
    G, edge_attr=None, node_attrs=None, iterations=3, digest_size=16
):
    def weisfeiler_lehman_step(G, labels, edge_attr=None):
        """ 
        Apply neighborhood aggregation to each node
        in the graph.
        Computes a dictionary with labels for each node.
        """
        new_labels = {}
        for node in G.nodes():
            label = _neighborhood_aggregate(G, node, labels, edge_attr=edge_attr)
            new_labels[node] = _hash_label(label, digest_size)
        return new_labels

    # set initial node labels
    node_labels = _init_node_labels(G, node_attrs)

    for _ in range(iterations):
        node_labels = weisfeiler_lehman_step(G, node_labels, edge_attr=edge_attr)
    return node_labels
