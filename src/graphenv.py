import itertools
import enum
import math
import random
from copy import deepcopy
from dataclasses import dataclass, field

import networkx as nx
import igraph as ig
import numpy as np

from matplotlib import colormaps
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch_geometric.data as gd
from torch_geometric.transforms import AddRandomWalkPE

from rdkit import Chem

from collections import Counter
from src.graph_utils import full_random_walk_pe, weisfeiler_lehman_graph_hash


NODE_COLORMAP = colormaps["Set3"]
EDGE_COLORMAP = colormaps["tab10"]


@dataclass
class GraphState:
    node_types: list[int] = field(default_factory=list)
    edge_types: list[int] = field(default_factory=list)
    edge_list: list[tuple[int, int]] = field(
        default_factory=list
    )  # assume edge[0] < edge[1]

    def __post_init__(self):
        self.num_nodes: int = len(self.node_types)
        self.num_edges: int = len(self.edge_list)
        self._edge_set: set[tuple[int, int]] = set(self.edge_list)
        self._set_degree()

    def _set_degree(self):
        degree = [0] * self.num_nodes
        for i, j in self.edge_list:
            degree[i] += 1
            degree[j] += 1
        self.degree = degree

    def count_automorphisms(self):
        '''
        Computes |Aut(G)|.
        '''
        if self.num_nodes <= 1:
            return 1
        g = ig.Graph()
        g.add_vertices(self.num_nodes)
        g.add_edges(self.edge_list)
        return g.count_automorphisms_vf2(
            color=self.node_types, edge_color=self.edge_types
        )
    
    def count_automorphisms_wl(self):
        '''
        Approximate automorphism counter using the Weisfeiler-Lehman (WL) algorithm.
        If the final coloring has distinct color classes, and class i has size k, 
        then an approximation is computed as \prod_{i=1}^{k} s_i. 
        This assumes that nodes within each class can be permuted freely, which is an upper bound of |Aut(G)|.
        '''
        wl = weisfeiler_lehman_graph_hash(self.to_nx(), 'edge_type', ['node_type'])
        facts = [math.factorial(cnt) for cnt in Counter(wl.values()).values()]
        return np.prod(facts)

    def get_automorphisms(self):
        if self.num_nodes <= 1:
            return [0] * self.num_nodes
        g = ig.Graph()
        g.add_vertices(self.num_nodes)
        g.add_edges(self.edge_list)
        return g.get_automorphisms_vf2(
            color=self.node_types, edge_color=self.edge_types
        )

    def _relabel(self, edge, index):
        if edge[0] > index:
            return edge[0] - 1, edge[1] - 1
        if edge[1] > index:
            return edge[0], edge[1] - 1
        return edge

    def add_node(self, node_type: int):
        self.node_types.append(node_type)
        self.degree.append(0)
        self.num_nodes += 1

    def add_edge(self, i: int, j: int, edge_type: int):
        edge = (i, j) if i < j else (j, i)
        assert edge not in self._edge_set
        self.edge_list.append(edge)
        self.edge_types.append(edge_type)
        self._edge_set.add(edge)
        self.num_edges += 1
        self.degree[i] += 1
        self.degree[j] += 1

    def remove_node(self, source):
        no_remove = [i for i, edge in enumerate(self.edge_list) if source not in edge]
        self.node_types.pop(source)
        self.edge_list = [self._relabel(self.edge_list[i], source) for i in no_remove]
        self.edge_types = [self.edge_types[i] for i in no_remove]
        self.__post_init__()

    def remove_edge(self, i, j):
        edge = (i, j) if i < j else (j, i)
        self._edge_set.remove(edge)
        remove_idx = self.edge_list.index(edge)
        self.edge_list.pop(remove_idx)
        self.edge_types.pop(remove_idx)
        self.num_edges -= 1
        self.degree[i] -= 1
        self.degree[j] -= 1

    def to_nx(self):
        graph = nx.from_edgelist(self.edge_list)
        # add missing nodes and set attributes
        for i, node_type in zip(range(self.num_nodes), self.node_types):
            graph.add_node(i, node_type=node_type)
        edge_attr = {
            edge: {"edge_type": self.edge_types[i]}
            for i, edge in enumerate(self.edge_list)
        }
        nx.set_edge_attributes(graph, edge_attr)
        return graph

    def draw(self, figsize=(3, 3), with_labels=True):
        g = self.to_nx()
        node_color = [NODE_COLORMAP(g.nodes[i]["node_type"]) for i in g.nodes]
        edge_color = [EDGE_COLORMAP(g.edges[i]["edge_type"]) for i in g.edges]
        plt.figure(figsize=figsize)
        return nx.draw(
            g,
            node_color=node_color,
            edge_color=edge_color,
            with_labels=with_labels,
        )


class ActionType(enum.Enum):
    Stop = enum.auto()
    AddNode = enum.auto()
    AddEdge = enum.auto()
    RemoveNode = enum.auto()
    RemoveEdge = enum.auto()


@dataclass
class GraphAction:
    type: ActionType = None
    source: int = None
    target: int = None
    node_type: int = None
    edge_type: int = None


@dataclass
class Neighbor:
    '''
    Parents or Children states in DAG, with actions that lead to corresponding states.
    '''
    states: list[GraphState] = field(default_factory=list)
    actions: list[GraphAction] = field(default_factory=list)

    def __len__(self):
        return len(self.states)


@dataclass
class Transition:
    states: list[GraphState] = field(default_factory=list)
    action: GraphAction = None
    torch_graphs: list[gd.Data] = field(default_factory=list)
    fwd_action_idx: int = None
    bck_action_idx: int = None
    fwd_equivalents: list[int] = field(default_factory=list)
    bck_equivalents: list[int] = field(default_factory=list)
    bck_log_prob: float = None
    log_reward: float = 0.0
    last_log_sym: float = 0.0
    log_sym_ratio: float = 0.0
    is_terminal: bool = False


@dataclass
class Trajectory:
    """
    Complete trajectory has N elements in states, actions, torch_graphs and fwd_action_idxs,
    while there are (N - 1) elements in bck_action_idxs and bck_log_probs.
    """

    states: list[GraphState] = field(default_factory=list)
    actions: list[GraphAction] = field(default_factory=list)
    torch_graphs: list[gd.Data] = field(default_factory=list)
    fwd_action_idxs: list[int] = field(default_factory=list)
    bck_action_idxs: list[int] = field(default_factory=list)
    fwd_equivalents: list[list[int]] = field(default_factory=list)
    bck_equivalents: list[list[int]] = field(default_factory=list)
    bck_log_probs: list[float] = field(default_factory=list)
    log_reward: float = None
    log_symmetries: list[float] = field(default_factory=list)
    last_log_sym: float = 0.0

    def to_transitions(self):
        transitions = []
        traj_len = len(self.states)
        for i in range(traj_len):
            if i < traj_len - 1:
                trantition = Transition(
                    states=self.states[i : i + 2],
                    action=self.actions[i],
                    torch_graphs=self.torch_graphs[i : i + 2],
                    fwd_action_idx=self.fwd_action_idxs[i],
                    bck_action_idx=(
                        self.bck_action_idxs[i] if self.bck_action_idxs else None
                    ),
                    bck_equivalents=(
                        self.bck_equivalents[i] if self.bck_equivalents else None
                    ),
                    bck_log_prob=self.bck_log_probs[i] if self.bck_log_probs else None,
                    fwd_equivalents=(
                        self.fwd_equivalents[i] if self.fwd_equivalents else None
                    ),
                    log_reward=0.0,
                    last_log_sym=0.0,
                    log_sym_ratio=(
                        self.log_symmetries[i + 1] - self.log_symmetries[i]
                        if self.log_symmetries
                        else 0.0
                    ),
                    is_terminal=False,
                )
            else:
                trantition = Transition(
                    states=[self.states[-1]] * 2,
                    action=self.actions[i],
                    torch_graphs=[self.torch_graphs[-1]] * 2,
                    fwd_action_idx=self.fwd_action_idxs[i],
                    bck_action_idx=0,
                    bck_equivalents=[0],
                    bck_log_prob=0.0,
                    fwd_equivalents=(
                        self.fwd_equivalents[i] if self.fwd_equivalents else None
                    ),
                    log_reward=self.log_reward,
                    last_log_sym=self.last_log_sym,
                    log_sym_ratio=0.0,
                    is_terminal=True,
                )
            transitions.append(trantition)
        return transitions


def _reward_num_cycles(state: GraphState) -> float:
    nx_graph = state.to_nx()
    reward = 1.0 + float(len(nx.cycle_basis(nx_graph)))
    return math.log(max(reward, 1e-8))

def _reward_uniform(state: GraphState) -> float:
    return 0.0

def _colored_n_clique_reward(state, n=4):
    '''
    Retrieved from https://github.com/recursionpharma/gflownet/tree/cliques_env
    '''
    g = state.to_nx()
    cliques = list(nx.algorithms.clique.find_cliques(g))
    # The number of cliques each node belongs to
    num_cliques = np.bincount(sum(cliques, []))
    colors = {i: g.nodes[i]["node_type"] for i in g.nodes}
    color_match = lambda c: np.bincount([colors[i] for i in c]).max() >= n - 1
    cliques_match = [
        float(len(i) == n) * (1 if color_match(i) else 0.5) for i in cliques
    ]
    return np.maximum(np.sum(cliques_match) - np.sum(num_cliques) + len(g) - 1, -10)


_atoms = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
_bonds = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
]

def _smiles_hash(state: GraphState) -> str:
    '''
    Hash a graph by pretending node/edge types represents atom/bond types of a molecule, 
    and converting it to SMILES representation.
    This is used to evaluate exact terminal probabilities.
    This hashing method may fail in some cases, but worked for our experiments.
    '''
    m = Chem.RWMol()
    for i in range(state.num_nodes):
        m.AddAtom(Chem.Atom(_atoms[state.node_types[i]]))
    for i, (u, v) in enumerate(state.edge_list):
        m.AddBond(u, v, order=_bonds[state.edge_types[i]])
    return Chem.MolToSmiles(m)


class GraphEnv:

    def __init__(
        self,
        max_nodes=10,
        max_edges=10,
        max_degree=4,
        num_node_types=1,
        num_edge_types=1,
        start_from_empty=False,
        reward_name="num_cycles",
        random_walk_length=4,
        add_degree=True,
        add_clustering_coef=True,
        add_shortest_path_length=True,
        offdiag_pe_length=0,
        **kargs
    ):
        '''
        Graph-building environment. 
        Implementation ideas are borrowed from recursionpharma/gflownet [1], but largely adapted for our purposes.
        
        [1] https://github.com/recursionpharma/gflownet/blob/trunk/src/gflownet/envs/graph_building_env.py
        '''
        self.max_nodes = max_nodes
        if max_edges is None:
            max_edges = max_nodes * (max_nodes - 1) // 2
        self.max_edges = max_edges
        if max_degree is None:
            max_degree = max_nodes - 1
        self.max_degree = max_degree
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.start_from_empty = start_from_empty
        self.reward_name = reward_name

        if reward_name == "num_cycles":
            self._reward_fn = _reward_num_cycles
        elif reward_name == "uniform":
            self._reward_fn = _reward_uniform
        elif reward_name == "cliques":
            self._reward_fn = _colored_n_clique_reward
        else:
            raise ValueError(f"reward_name={reward_name}")

        self.randomwalk_pe = AddRandomWalkPE(random_walk_length, "rwpe")
        self.add_degree = add_degree
        self.add_clustering_coef = add_clustering_coef
        self.add_shortest_path_length = add_shortest_path_length
        self.offdiag_pe_length = offdiag_pe_length
        self.node_dim = (
            self.num_node_types
            + random_walk_length
            + (add_degree and self.max_degree + 1)
            + add_clustering_coef
        )
        self.edge_dim = self.num_edge_types

        self.first_log_sym = math.log(self.new().count_automorphisms())

    def new(self):
        return GraphState(node_types=[0] * (not self.start_from_empty))

    def hash(self, state: GraphState) -> str:
        return _smiles_hash(state)

    def stop_action(self):
        return GraphAction(ActionType.Stop)

    def step(self, state: GraphState, action: GraphAction) -> tuple[GraphState, bool]:
        next_state = deepcopy(state)
        if action.type == ActionType.AddNode:
            assert action.node_type < self.num_node_types
            next_state.add_node(action.node_type)
            if action.source is not None:
                target = next_state.num_nodes - 1
                next_state.add_edge(action.source, target, action.edge_type)
        elif action.type == ActionType.AddEdge:
            assert action.edge_type < self.num_edge_types
            next_state.add_edge(action.source, action.target, action.edge_type)
        elif action.type == ActionType.RemoveNode:
            next_state.remove_node(action.source)
        elif action.type == ActionType.RemoveEdge:
            next_state.remove_edge(action.source, action.target)
        elif action.type == ActionType.Stop:
            pass
        else:
            raise ValueError()
        return next_state

    def reverse_action(self, state, action):
        if action.type == ActionType.RemoveEdge:
            action = GraphAction(
                ActionType.AddEdge,
                source=action.source,
                target=action.target,
                edge_type=action.edge_type,
            )
        elif action.type == ActionType.RemoveNode:
            if state.num_nodes == 1:
                action = GraphAction(ActionType.AddNode, node_type=state.node_types[0])
            else:
                for edge_type, edge in zip(state.edge_types, state.edge_list):
                    if action.source in edge:
                        target = edge[1] if edge[0] == action.source else edge[0]
                        break
                node_type = state.node_types[action.source]
                source = target - int(action.source < target)
                action = GraphAction(
                    ActionType.AddNode, source, node_type=node_type, edge_type=edge_type
                )
        elif action.type == ActionType.AddEdge:
            action = GraphAction(
                ActionType.RemoveEdge,
                source=action.source,
                target=action.target,
                edge_type=action.edge_type,
            )
        elif action.type == ActionType.AddNode:
            if action.source is None:
                action = GraphAction(ActionType.RemoveNode, source=0)
            else:
                action = GraphAction(ActionType.RemoveNode, source=state.num_nodes)
        elif action.type == ActionType.Stop:
            action = action
        return action

    def log_reward(self, state: GraphState) -> float:
        return self._reward_fn(state)

    def parents_actions(self, state: GraphState) -> list[GraphAction]:
        actions = []

        num_init_node_type = sum([x == 0 for x in state.node_types])

        # ActionType.RemoveNode
        for source in range(state.num_nodes):
            if state.degree[source] <= 1:
                if (
                    self.start_from_empty
                    or (state.node_types[source] != 0)
                    or (num_init_node_type > 1)
                ):
                    action = GraphAction(ActionType.RemoveNode, source=source)
                    actions.append(action)

        # ActionType.RemoveEdge
        for idx, (u, v) in enumerate(state.edge_list):
            if state.degree[u] > 1 and state.degree[v] > 1:
                action = GraphAction(
                    ActionType.RemoveEdge,
                    source=u,
                    target=v,
                    edge_type=state.edge_types[idx],
                )
                new_state = self.step(state, action)
                new_graph = new_state.to_nx()
                if nx.algorithms.is_connected(new_graph):
                    actions.append(action)
        return actions

    def parents(self, state: GraphState) -> Neighbor:
        actions = self.parents_actions(state)
        next_states = [self.step(state, act) for act in actions]
        parents = Neighbor(next_states, actions)
        return parents

    def children_actions(self, state: GraphState) -> list[GraphAction]:

        if state.num_nodes == 0:
            actions = []
            for node_type in range(self.num_node_types):
                actions.append(GraphAction(ActionType.AddNode, node_type=node_type))
            return actions

        actions = [self.stop_action()]

        if state.num_edges < self.max_edges:
            allowable_nodes = [
                i for i, deg in enumerate(state.degree) if deg < self.max_degree
            ]

            # AddNode
            if state.num_nodes < self.max_nodes:
                for i in allowable_nodes:
                    for node_type in range(self.num_node_types):
                        for edge_type in range(self.num_edge_types):
                            actions.append(
                                GraphAction(
                                    ActionType.AddNode,
                                    source=i,
                                    node_type=node_type,
                                    edge_type=edge_type,
                                )
                            )
            # AddEdge
            for a, b in itertools.combinations(allowable_nodes, 2):
                if (a, b) not in state._edge_set:
                    for edge_type in range(self.num_edge_types):
                        actions.append(
                            GraphAction(
                                ActionType.AddEdge,
                                source=a,
                                target=b,
                                edge_type=edge_type,
                            )
                        )
        return actions

    def children(self, state: GraphState) -> Neighbor:
        actions = self.children_actions(state)
        next_states = [self.step(state, act) for act in actions]
        children = Neighbor(next_states, actions)
        return children

    def equivalent_action_idxs(self, neighbor: Neighbor, action_idx: int) -> list[int]:
        child = neighbor.states[action_idx]
        equivalent_idxs = [action_idx]
        child_graph = child.to_nx()
        for i in range(len(neighbor)):
            if i == action_idx:
                continue
            other = neighbor.states[i]
            if nx.is_isomorphic(
                child_graph, other.to_nx(), lambda a, b: a == b, lambda a, b: a == b
            ):
                equivalent_idxs.append(i)
        return equivalent_idxs

    def equivalent_action_idxs_using_pe(self, neighbor: Neighbor, action_idx: int, pe: torch.Tensor) -> list[int]:
        action = neighbor.actions[action_idx]
        equivalent_idxs = [action_idx]

        if action.type == ActionType.Stop:
            return equivalent_idxs
        elif action.type.name.endswith("Node"):
            action_pe = pe[action.source]
        elif action.type.name.endswith("Edge"):
            action_pe = pe[action.source] + pe[action.target]
        else:
            raise ValueError()
            
        for i in range(len(neighbor)):
            if i == action_idx:
                continue
            other = neighbor.actions[i]
            if (action.type == other.type) and (action.node_type == other.node_type) and (action.edge_type == other.edge_type):
                if other.type.name.endswith("Node"):
                    other_pe = pe[other.source]
                elif other.type.name.endswith("Edge"):
                    other_pe = pe[other.source] + pe[other.target]

                if torch.allclose(action_pe, other_pe):
                    equivalent_idxs.append(i)
        return equivalent_idxs

    def random_state(self):
        state = self.new()
        while 1:
            action = random.choice(self.children_actions(state))
            if action.type == ActionType.Stop:
                break
            else:
                state = self.step(state, action)
        return state

    def featurize(self, states: list[GraphState], set_backward=False):
        output = {"graph": [], "children": [], "parents": []}

        for state in states:
            torch_graph = self.state_to_torch_graph(state)

            children = self.children(state)
            self._set_fwd_action_attr(torch_graph, children)
            output["children"].append(children)

            if self.add_shortest_path_length:
                torch_graph.fwd_path_lens = self._get_shortest_path_length(
                    state, children
                )
            if self.offdiag_pe_length:
                torch_graph.offdiag_pe = self._get_offdiag_pe(
                    torch_graph, children
                )

            if set_backward:
                parents = self.parents(state)
                self._set_bck_action_attr(torch_graph, parents)
                output["parents"].append(parents)
                if self.add_shortest_path_length:
                    torch_graph.bck_path_lens = self._get_shortest_path_length(
                        state, parents
                    )

            output["graph"].append(torch_graph)
        return output

    def state_to_torch_graph(self, state: GraphState) -> gd.Data:
        node_types = torch.tensor([i for i in state.node_types], dtype=torch.long)
        edge_types = torch.tensor([i for i in state.edge_types], dtype=torch.long)
        edge_index = [e for i, j in state.edge_list for e in [(i, j), (j, i)]]
        edge_index = (
            torch.tensor(edge_index, dtype=torch.long).reshape(-1, 2).t().contiguous()
        )
        torch_graph = gd.Data(
            node_types=node_types,
            edge_types=edge_types,
            edge_index=edge_index,
        )
        self._set_graph_features(torch_graph, state)
        return torch_graph

    def _get_shortest_path_length(self, state: GraphState, neighbor: Neighbor):
        all_lens = dict(nx.shortest_path_length(state.to_nx()))
        lens = []
        for action in neighbor.actions:
            if action.type in {ActionType.AddEdge, ActionType.RemoveEdge}:
                # assign 0 to disconnected nodes
                lens.append(all_lens[action.source].get(action.target, 0))
        return torch.tensor(lens, dtype=torch.float)
    
    def _get_offdiag_pe(self, torch_graph: gd.Data, neighbor: Neighbor):
        pe = full_random_walk_pe(torch_graph, self.offdiag_pe_length)
        sources, targets = [], []
        for action in neighbor.actions:
            if action.type in {ActionType.AddEdge, ActionType.RemoveEdge}:
                sources.append(action.source)
                targets.append(action.target)
        return pe[sources, targets]

    def _set_graph_features(self, torch_graph: gd.Data, state: GraphState):
        x = F.one_hot(torch_graph.node_types, num_classes=self.num_node_types).float()
        if self.randomwalk_pe.walk_length > 0:
            rwpe = self.randomwalk_pe(torch_graph).rwpe
            x = torch.cat([x, rwpe], dim=1)
        if self.add_degree:
            degree = torch.tensor(state.degree, dtype=torch.long)
            deg = F.one_hot(degree, num_classes=self.max_degree + 1).float()
            x = torch.cat([x, deg], dim=1)
        if self.add_clustering_coef:
            clustering = nx.clustering(state.to_nx())
            coef = [clustering[i] for i in range(len(clustering))]
            coef = torch.tensor(coef, dtype=torch.float)[:, None]
            x = torch.cat([x, coef], dim=1)
        if x.shape[0] == 0:
            x = torch.zeros(1, x.shape[1], dtype=torch.float)
        edge_types = torch_graph.edge_types.repeat_interleave(2, dim=0)
        edge_attr = F.one_hot(edge_types, num_classes=self.num_edge_types).float()
        torch_graph.x = x
        torch_graph.edge_attr = edge_attr
        torch_graph.node_types
        torch_graph.edge_types

    def _set_fwd_action_attr(self, torch_graph: gd.Data, children: Neighbor):
        add_node_index = []
        add_node_type = []
        add_edge_index = []
        add_edge_type = []
        has_stop = False
        for action in children.actions:
            if action.type == ActionType.AddNode:
                source, edge_type = (action.source or 0, action.edge_type or 0)
                add_node_index.append(source)
                add_node_type.append(action.node_type * self.num_edge_types + edge_type)
            elif action.type == ActionType.AddEdge:
                edge = (action.source, action.target)
                add_edge_index.append(edge)
                add_edge_type.append(action.edge_type)
            elif action.type == ActionType.Stop:
                has_stop = True

        add_node_index = torch.tensor(add_node_index, dtype=torch.long)
        add_node_type = torch.tensor(add_node_type, dtype=torch.long)
        add_node_mask = F.one_hot(
            add_node_type, num_classes=self.num_node_types * self.num_edge_types
        ).bool()

        add_edge_index = (
            torch.tensor(add_edge_index, dtype=torch.long)
            .reshape(-1, 2)
            .t()
            .contiguous()
        )
        add_edge_type = torch.tensor(add_edge_type, dtype=torch.long)
        add_edge_mask = F.one_hot(add_edge_type, num_classes=self.num_edge_types).bool()

        torch_graph.add_node_index = add_node_index
        torch_graph.add_edge_index = add_edge_index
        torch_graph.add_node_mask = add_node_mask
        torch_graph.add_edge_mask = add_edge_mask
        torch_graph.stop_mask = torch.tensor([has_stop], dtype=torch.bool)

    def _set_bck_action_attr(self, torch_graph: gd.Data, parents: Neighbor):
        del_node_index = []
        del_edge_index = []
        for action in parents.actions:
            if action.type == ActionType.RemoveNode:
                del_node_index.append(action.source)
            elif action.type == ActionType.RemoveEdge:
                del_edge_index.append((action.source, action.target))

        torch_graph.del_node_index = torch.tensor(del_node_index, dtype=torch.long)
        torch_graph.del_edge_index = (
            torch.tensor(del_edge_index, dtype=torch.long)
            .reshape(-1, 2)
            .t()
            .contiguous()
        )

    def collate(self, torch_graphs, set_backward=False):
        batch = gd.Batch.from_data_list(
            torch_graphs, follow_batch=["add_edge_index", "del_edge_index"]
        )

        stop_batch = batch.stop_mask.nonzero().flatten()
        add_node_batch = batch.batch[batch.add_node_index]
        add_edge_batch = batch.add_edge_index_batch
        batch.fwd_logit_batch = torch.cat(
            [stop_batch, add_node_batch, add_edge_batch], dim=0
        )
        del batch.add_edge_index_batch
        del batch.add_edge_index_ptr

        if set_backward:
            del_node_batch = batch.batch[batch.del_node_index]
            del_edge_batch = batch.del_edge_index_batch
            batch.bck_logit_batch = torch.cat([del_node_batch, del_edge_batch])
            del batch.del_edge_index_batch
            del batch.del_edge_index_ptr
        return batch

    def unif_backward_sample(self, state: GraphState, set_equivalents: bool = False):
        num_steps = state.num_edges + self.start_from_empty
        states = [self.new()] + [None] * (num_steps - 1) + [state]
        fwd_actions = [None] * num_steps + [self.stop_action()]
        torch_graphs = [None] * (num_steps + 1)
        fwd_action_idxs = [None] * (num_steps + 1)
        bck_action_idxs = [None] * num_steps
        bck_log_probs = [None] * num_steps
        fwd_equivalents = [None] * num_steps if set_equivalents else []
        bck_equivalents = [None] * num_steps if set_equivalents else []

        for t in reversed(range(num_steps)):
            features = self.featurize([state], set_backward=True)
            children = features["children"][0]
            parents = features["parents"][0]
            torch_graph = features["graph"][0]
            i = random.randint(0, len(parents) - 1)
            bck_action = parents.actions[i]
            fwd_action = self.reverse_action(state, bck_action)
            states[t] = parents.states[i]
            fwd_actions[t] = fwd_action
            torch_graphs[t + 1] = torch_graph
            fwd_action_idxs[t + 1] = children.actions.index(fwd_actions[t + 1])
            bck_action_idxs[t] = i
            bck_log_probs[t] = -math.log(len(parents))
            state = states[t]
            if set_equivalents:
                bck_equivalents[t] = self.equivalent_action_idxs(parents, i)
                bck_log_probs[t] = math.log(len(bck_equivalents[t])) + bck_log_probs[t]
                fwd_equivalents[t] = self.equivalent_action_idxs(
                    children, fwd_action_idxs[t + 1]
                )

        features = self.featurize([state], set_backward=True)
        torch_graphs[0] = features["graph"][0]
        fwd_action_idxs[0] = features["children"][0].actions.index(fwd_actions[0])
        if set_equivalents:
            fwd_equivalents[0] = self.equivalent_action_idxs(
                features["children"][0], fwd_action_idxs[0]
            )
        traj = Trajectory(
            states=states,
            actions=fwd_actions,
            torch_graphs=torch_graphs,
            fwd_action_idxs=fwd_action_idxs,
            bck_action_idxs=bck_action_idxs,
            bck_log_probs=bck_log_probs,
            fwd_equivalents=fwd_equivalents,
            bck_equivalents=bck_equivalents,
        )
        return traj

    def terminal_states_info(self):
        state = self.new()
        smiles = self.hash(state)
        depth = 0
        info = {
            smiles: {"num_trajs": 1, "state": state, "reward": None, "depth": depth}
        }
        frontier = [(state, smiles)]
        while frontier:
            next_frontier = []
            for state, parent_smiles in frontier:
                children = self.children(state)
                for child, action in zip(children.states, children.actions):
                    if action.type.name == "Stop":
                        info[parent_smiles]["reward"] = np.exp(self.log_reward(child))
                    else:
                        child_smiles = self.hash(child)
                        if child_smiles not in info:
                            info[child_smiles] = {
                                "num_trajs": info[parent_smiles]["num_trajs"],
                                "state": child,
                                "reward": None,
                                "depth": depth + 1,
                            }
                            next_frontier.append((child, child_smiles))
                        else:
                            info[child_smiles]["num_trajs"] += info[parent_smiles][
                                "num_trajs"
                            ]
            frontier = next_frontier
            depth += 1
        return info


class IllustrativeEnv(GraphEnv):
    def __init__(
        self,
        max_nodes=5,
        max_degree=None,
        random_walk_length=4,
        add_degree=True,
        add_clustering_coef=True,
        add_shortest_path_length=True,
        offdiag_pe_length=0,
        **kargs
    ):
        '''
        Graph-building environment with disconnected nodes as the initial state, where only AddEnde actions are allowed.
        This is used for the illustrative purpose.
        '''
        super().__init__(
            max_nodes=max_nodes,
            max_edges=None,
            max_degree=max_degree,
            reward_name="uniform",
            random_walk_length=random_walk_length,
            add_degree=add_degree,
            add_clustering_coef=add_clustering_coef,
            add_shortest_path_length=add_shortest_path_length,
            offdiag_pe_length=offdiag_pe_length
        )

    def new(self):
        return GraphState(node_types=[0] * self.max_nodes)

    def parents_actions(self, state: GraphState) -> list[GraphAction]:
        actions = []
        for idx, (u, v) in enumerate(state.edge_list):
            if state.degree[u] > 0 and state.degree[v] > 0:
                actions.append(
                    GraphAction(
                        ActionType.RemoveEdge,
                        source=u,
                        target=v,
                        edge_type=state.edge_types[idx],
                    )
                )
        return actions

    def children_actions(self, state: GraphState) -> list[GraphAction]:
        actions = []

        if nx.is_connected(state.to_nx()):
            actions.append(self.stop_action())

        if state.num_edges < self.max_edges:
            allowable_nodes = [
                i for i, deg in enumerate(state.degree) if deg < self.max_degree
            ]

            for a, b in itertools.combinations(allowable_nodes, 2):
                if (a, b) not in state._edge_set:
                    for edge_type in range(self.num_edge_types):
                        actions.append(
                            GraphAction(
                                ActionType.AddEdge,
                                source=a,
                                target=b,
                                edge_type=edge_type,
                            )
                        )
        return actions


class MolEnv(GraphEnv):
    atoms = _atoms
    bonds = _bonds
    num_node_types = len(atoms)
    num_edge_types = len(bonds)

    def __init__(
        self,
        max_nodes=10,
        num_node_types=None,
        num_edge_types=None,
        reward_name="plogp",
        reward_exponent=1.0,
        random_walk_length=4,
        add_degree=True,
        add_clustering_coef=True,
        add_shortest_path_length=True,
        **kargs
    ):
        '''
        Molecule generation environment. This is not used in the paper.
        '''
        pt = Chem.GetPeriodicTable()
        self.max_atom_valence = {a: max(pt.GetValenceList(a)) for a in self.atoms}
        max_degree = max(self.max_atom_valence.values())
        if num_node_types:
            self.num_node_types = num_node_types
            self.atoms = self.atoms[:num_node_types]

        if num_edge_types:
            self.num_edge_types = num_edge_types
            self.bonds = self.bonds[:num_edge_types]

        super().__init__(
            max_nodes,
            None,
            max_degree,
            random_walk_length=random_walk_length,
            add_degree=add_degree,
            add_clustering_coef=add_clustering_coef,
            add_shortest_path_length=add_shortest_path_length,
        )

        self.reward_name = reward_name
        self.reward_exponent = reward_exponent

        if reward_name in set(self.atoms):
            self._reward_fn = lambda mol: math.log(
                0.1 + sum([reward_name == atom.GetSymbol() for atom in mol.GetAtoms()])
            )
        else:
            raise ValueError(f"reward_name={reward_name}")

    def _get_atom_valence(self, state: GraphState) -> list[int]:
        valence = [0] * state.num_nodes
        for t, (u, v) in zip(state.edge_types, state.edge_list):
            valence[u] += t + 1
            valence[v] += t + 1
        return valence

    def state_to_mol(self, state: GraphState) -> Chem.Mol:
        m = Chem.RWMol()
        for t in state.node_types:
            atom_str = self.atoms[t]
            atom = Chem.Atom(atom_str)
            m.AddAtom(atom)
        for t, (u, v) in zip(state.edge_types, state.edge_list):
            m.AddBond(u, v, order=self.bonds[t])
        return m

    def mol_to_state(self, mol):
        Chem.Kekulize(mol, clearAromaticFlags=True)
        node_types = []
        for atom in mol.GetAtoms():
            atom_str = atom.GetSymbol()
            atom_idx = self.atoms.index(atom_str)
            node_types.append(atom_idx)
        edge_types, edge_list = [], []
        for bond in mol.GetBonds():
            edge_types.append(self.bonds.index(bond.GetBondType()))
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge = (i, j) if i < j else (j, i)
            edge_list.append(edge)
        return GraphState(node_types, edge_types, edge_list)

    def hash(self, state: GraphState) -> str:
        m = self.state_to_mol(state)
        return Chem.MolToSmiles(m)

    def log_reward(self, state: GraphState) -> float:
        mol = self.state_to_mol(state)
        Chem.SanitizeMol(mol)
        log_reward = self._reward_fn(mol)
        return self.reward_exponent * log_reward

    def children_actions(self, state: GraphState) -> list[GraphAction]:
        actions = []
        if state.num_nodes == 0:
            for node_type in range(self.num_node_types):
                actions(GraphAction(ActionType.AddNode, node_type=node_type))
            return actions

        actions.append(self.stop_action())

        if state.num_edges < self.max_edges:
            atom_val = self._get_atom_valence(state)
            impl_val = [
                self.max_atom_valence.get(self.atoms[t]) - atom_val[i]
                for i, t in enumerate(state.node_types)
            ]
            allowable_nodes = [i for i, v in enumerate(impl_val) if v > 0]

            # AddNode
            if state.num_nodes < self.max_nodes:
                for i in allowable_nodes:
                    for node_type in range(self.num_node_types):
                        max_val = self.max_atom_valence.get(self.atoms[node_type])
                        for edge_type in range(self.num_edge_types):
                            if edge_type < min(impl_val[i], max_val):
                                actions.append(
                                    GraphAction(
                                        ActionType.AddNode,
                                        source=i,
                                        node_type=node_type,
                                        edge_type=edge_type,
                                    )
                                )

            # AddEdge
            for a, b in itertools.combinations(allowable_nodes, 2):
                if (a, b) not in state._edge_set:
                    for edge_type in range(self.num_edge_types):
                        if edge_type < min(impl_val[a], impl_val[b]):
                            actions.append(
                                GraphAction(
                                    ActionType.AddEdge,
                                    source=a,
                                    target=b,
                                    edge_type=edge_type,
                                )
                            )
        return actions
