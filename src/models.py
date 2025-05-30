import torch
from torch import Tensor
from torch import nn
from torch_geometric.nn.conv import GPSConv, GINEConv
from torch_geometric.nn import global_add_pool
import torch_geometric.data as gd
from torch_scatter import (
    scatter_max,
    scatter_logsumexp,
    scatter_log_softmax,
    scatter_add,
)

from src.graphenv import GraphEnv


class GroupCategorical:

    _gumb = torch.distributions.Gumbel(0.0, 1.0)

    def __init__(self, logits: Tensor, batch: Tensor):
        assert len(logits) == len(batch)
        batch, sort_idx = torch.sort(batch, stable=True)
        self.logits = logits[sort_idx]
        self.device = logits.device
        counts = torch.bincount(batch)
        sizes = counts[counts != 0]
        self.batch = torch.repeat_interleave(sizes)
        self._offset = torch.cumsum(sizes, 0) - sizes

    def select(self, indices):
        return self.logits[self._offset + indices]

    def log_prob(self, value: Tensor = None):
        logprobs = scatter_log_softmax(self.logits, self.batch)
        if value is None:
            return logprobs
        return logprobs[value + self._offset]

    def logsumexp(self):
        return scatter_logsumexp(self.logits, self.batch)

    def subset_logsumexp(self, indices: Tensor, batch_subset: Tensor):
        offset = self._offset[batch_subset]
        logprobs = scatter_log_softmax(self.logits, self.batch)
        logprobs_subset = logprobs.gather(0, indices + offset)
        return scatter_logsumexp(logprobs_subset, index=batch_subset)

    @torch.no_grad()
    def sample(self, temperature: int = 1.0):
        """Sample from softmax probabilities using Gumbel-max trick."""
        gumbel = self._gumb.sample(self.logits.shape).to(self.device)
        scaled_logits = self.logits / temperature
        _, indices = scatter_max(scaled_logits + gumbel, self.batch)
        return indices - self._offset

    def entropy(self):
        logprobs = scatter_log_softmax(self.logits, self.batch)
        entropy = scatter_add(-logprobs * logprobs.exp(), self.batch)
        return entropy

    def to(self, device):
        self.device = device
        self.logits = self.logits.to(device)
        self.batch = self.batch.to(device)
        self._offset = self._offset.to(device)
        return self

    def detach(self):
        return self.__class__(self.logits.detach(), self.batch.detach())

    def __repr__(self):
        return f"{self.__class__.__name__}(size: {len(self.batch)})"


class GPS(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        embed_dim: int = 256,
        num_layers: int = 5,
        num_heads: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.node_emb = nn.Linear(node_dim, embed_dim, bias=False)
        self.edge_emb = nn.Linear(edge_dim, embed_dim, bias=False)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            conv = GPSConv(
                embed_dim,
                GINEConv(net),
                norm="layer_norm",
                heads=num_heads,
                attn_type="multihead",
            )
            self.convs.append(conv)

    def forward(self, g: gd.Batch):
        x = self.node_emb(g.x)
        edge_attr = self.edge_emb(g.edge_attr)
        for conv in self.convs:
            x = conv(x, g.edge_index, g.batch, edge_attr=edge_attr)
        glob = global_add_pool(x, g.batch)
        return x, glob


class PolicyNet(nn.Module):
    shortest_path_length_dim = 8

    def __init__(
        self,
        env: GraphEnv,
        embed_dim: int = 256,
        num_layers: int = 5,
        num_heads: int = 4,
        mlp_num_hidden: int = 1
    ):
        super().__init__()
        self.env = env
        self.gnn = GPS(
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        self.embed_dim = embed_dim
        self.edge_embed_dim = (
            embed_dim + self.shortest_path_length_dim * env.add_shortest_path_length + env.offdiag_pe_length
        )
        self.add_node_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, env.num_node_types * env.num_edge_types),
        )
        self.add_edge_mlp = nn.Sequential(
            *[nn.Linear(self.edge_embed_dim, self.edge_embed_dim), nn.GELU()] * mlp_num_hidden,
            nn.Linear(self.edge_embed_dim, env.num_edge_types),
        )
        self.stop_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def add_backward_policy(self):
        self.del_node_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 1),
        )
        self.del_edge_mlp = nn.Sequential(
            nn.Linear(self.edge_embed_dim, self.edge_embed_dim),
            nn.GELU(),
            nn.Linear(self.edge_embed_dim, 1),
        )

    def add_state_flow(self):
        self.flow_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 1),
        )

    def add_logZ(self):
        self.logZ = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: gd.Batch, return_backward_dist=False, return_flow=False):
        node_feature, graph_feature = self.gnn(batch.to(self.device))
        row, col = batch.add_edge_index
        edge_feature = node_feature[row] + node_feature[col]
        if self.env.add_shortest_path_length:
            log_fwd_lens = (batch.fwd_path_lens + 1).log().unsqueeze(1) * torch.ones(
                self.shortest_path_length_dim, device=self.device
            )
            edge_feature = torch.cat([edge_feature, log_fwd_lens], dim=1)
        if self.env.offdiag_pe_length:
            edge_feature = torch.cat([edge_feature, batch.offdiag_pe], dim=1)

        stop_logits = self.stop_mlp(graph_feature).flatten()
        add_node_logits = self.add_node_mlp(node_feature[batch.add_node_index])
        add_edge_logits = self.add_edge_mlp(edge_feature)
        stop_logits = stop_logits[batch.stop_mask]
        add_node_logits = add_node_logits[batch.add_node_mask]
        add_edge_logits = add_edge_logits[batch.add_edge_mask]

        fwd_logits = torch.cat([stop_logits, add_node_logits, add_edge_logits], dim=0)
        fwd_action_dist = GroupCategorical(fwd_logits, batch.fwd_logit_batch)

        bck_action_dist = None
        if return_backward_dist:
            row, col = batch.del_edge_index
            edge_feature = node_feature[row] + node_feature[col]
            if self.env.add_shortest_path_length:
                log_bck_lens = (batch.bck_path_lens + 1).log().unsqueeze(
                    1
                ) * torch.ones(self.shortest_path_length_dim, device=self.device)
                edge_feature = torch.cat([edge_feature, log_bck_lens], dim=1)
            del_node_logits = self.del_node_mlp(
                node_feature[batch.del_node_index]
            ).flatten()
            del_edge_logits = self.del_edge_mlp(edge_feature).flatten()
            bck_logits = torch.cat([del_node_logits, del_edge_logits], dim=0)
            bck_action_dist = GroupCategorical(bck_logits, batch.bck_logit_batch)

        state_flow = None
        if return_flow:
            state_flow = self.flow_mlp(graph_feature).flatten()
        return fwd_action_dist, bck_action_dist, state_flow

    def num_params(self) -> int:
        return sum([param.numel() for param in self.parameters()])

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
