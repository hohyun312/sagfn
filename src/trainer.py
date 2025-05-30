import logging
import time
import math
import enum
import random
import numpy as np
from collections import defaultdict
from copy import deepcopy

import torch
from torch.utils.data import IterableDataset, DataLoader
from torch_scatter import scatter_add

from src.graphenv import GraphEnv, Trajectory, Transition
from src.models import PolicyNet
from src.utils import ModelProxy, get_logger
from src.graph_utils import random_walk_pe_with_node_types



class CorrectionMethod(enum.Enum):
    '''
    Correction methods implemented.
        1) Vanilla: no corrections.
        2) Transition: exact transition correction using multiple isomorphism tests.
        3) PE: approximate correction using positional encodings to identify orbits as proposed by Ma et al [1].
        4) RewardScaling: exact correction, which scales final rewards by |Aut(G)|.
        5) FlowScaling: exact correction, which scales backward action probabilities by symmetry ratio |Aut(G')|/|Aut(G)|.
            this is equivalent to RewardScaling when using the TB objective.
        
        [1] Ma et al., Baking Symmetry into GFlowNets, https://arxiv.org/abs/2406.05426
    '''
    Vanilla = enum.auto()
    Transition = enum.auto()
    PE = enum.auto()
    RewardScaling = enum.auto()
    FlowScaling = enum.auto()


class ReplayBuffer:
    def __init__(self, buffer_size):
        '''
        Simple replay buffer for storing trajectories or transitions.
        '''
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0

    @classmethod
    def from_data(cls, data):
        buffer = cls(len(data))
        buffer.push_many(data)
        return buffer

    def push(self, obj):
        if self.buffer_size != 0:
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(None)
            self.buffer[self.position] = obj
            self.position = (self.position + 1) % self.buffer_size

    def push_many(self, objs):
        for obj in objs:
            self.push(obj)

    def sample(self, batch_size):
        out = []
        num_sample = min(len(self.buffer), batch_size)
        if num_sample > 0:
            indices = np.random.choice(len(self.buffer), num_sample)
            out = [self.buffer[i] for i in indices]
        return out

    def __len__(self):
        return len(self.buffer)


class GFNDataset(IterableDataset):
    def __init__(
        self,
        model: PolicyNet,
        env: GraphEnv,
        transition=False,
        train_size=64,
        sampling_size=32,
        sampling_freq=8,
        buffer_size=0,
        exploration_epsilon=0.0,
        unif_backward=True,
        correction_method: CorrectionMethod = CorrectionMethod.Vanilla,
    ):

        if not transition:
            # CorrectionMethod.FlowScaling only works for DB
            assert correction_method != CorrectionMethod.FlowScaling
        self.model = model
        self.env = env
        self.transition = transition
        self.train_size = train_size
        self.sampling_size = sampling_size
        self.sampling_freq = sampling_freq
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.exploration_epsilon = exploration_epsilon
        self.unif_backward = unif_backward
        self.correction_method = correction_method
        self.num_sampled = 0
        self.pe_length = 8

    def __iter__(self):
        while True:
            online_data = []
            if self.num_sampled % self.sampling_freq == 0:
                sample_start = time.time()
                online_data = self.sample_trajectories(
                    self.sampling_size,
                    self.exploration_epsilon,
                )
                self.set_bck_action_idxs(online_data)
                self.set_symmetries(online_data)
                sampling_time = time.time() - sample_start
                self.set_log_reward(online_data)
                self.num_sampled += 1
                if self.transition:
                    online_data = sum([x.to_transitions() for x in online_data], [])
                self.replay_buffer.push_many(online_data)

            buffer_data = self.replay_buffer.sample(self.train_size - len(online_data))
            all_data = online_data + buffer_data
            if self.transition:
                batch = self.make_transition_batch(all_data)
            else:
                batch = self.make_trajectory_batch(all_data)
            batch.sampling_time = sampling_time
            yield batch

    def set_bck_action_idxs(self, trajectories: list[Trajectory]):
        for traj in trajectories:
            bck_actions = [
                self.env.reverse_action(s, a)
                for s, a in zip(traj.states[:-1], traj.actions[:-1])
            ]
            parents_lst = [self.env.parents(s) for s in traj.states[1:]]
            bck_action_idxs = [
                parents.actions.index(action)
                for parents, action in zip(parents_lst, bck_actions)
            ]
            traj.bck_action_idxs = bck_action_idxs

            if self.correction_method == CorrectionMethod.Transition:
                traj.bck_equivalents = [
                    self.env.equivalent_action_idxs(parents, idx)
                    for parents, idx in zip(parents_lst, bck_action_idxs)
                ]
            elif self.correction_method == CorrectionMethod.PE:
                pes = [random_walk_pe_with_node_types(g, self.pe_length) for g in traj.torch_graphs[1:]]
                traj.bck_equivalents = [
                    self.env.equivalent_action_idxs_using_pe(parents, idx, pe)
                    for parents, idx, pe in zip(parents_lst, bck_action_idxs, pes)
                ]
            if self.unif_backward:
                traj.bck_log_probs = [
                    -math.log(len(parents)) for parents in parents_lst
                ]
                if self.correction_method in {CorrectionMethod.Transition, CorrectionMethod.PE}:
                    traj.bck_log_probs = [
                        math.log(len(lst)) + x
                        for lst, x in zip(traj.bck_equivalents, traj.bck_log_probs)
                    ]

    def set_symmetries(self, trajectories: list[Trajectory]):
        if self.correction_method == CorrectionMethod.RewardScaling:
            for traj in trajectories:
                last_state = traj.states[-1]
                traj.last_log_sym = math.log(last_state.count_automorphisms())
        elif self.correction_method == CorrectionMethod.FlowScaling:
            for traj in trajectories:
                traj.log_symmetries = [
                    math.log(state.count_automorphisms()) for state in traj.states
                ]

    def set_log_reward(self, trajectories: list[Trajectory]):
        for traj in trajectories:
            last_state = traj.states[-1]
            traj.log_reward = self.env.log_reward(last_state)

    def make_trajectory_batch(self, trajectories: list[Trajectory]):
        batch = self.env.collate(
            sum([traj.torch_graphs for traj in trajectories], []),
            set_backward=not self.unif_backward,
        )
        batch.num_trajectories = len(trajectories)
        batch.log_rewards = torch.tensor(
            [traj.log_reward for traj in trajectories], dtype=torch.float
        )
        trajectory_lens = torch.tensor(
            [len(traj.actions) for traj in trajectories], dtype=torch.long
        )
        batch.traj_fwd_batch = torch.repeat_interleave(trajectory_lens)
        batch.fwd_action_idxs = torch.tensor(
            sum([traj.fwd_action_idxs for traj in trajectories], []), dtype=torch.long
        )
        batch.traj_bck_batch = torch.repeat_interleave(trajectory_lens - 1)
        batch.bck_action_idxs = torch.tensor(
            sum([traj.bck_action_idxs for traj in trajectories], []), dtype=torch.long
        )
        if self.correction_method in {CorrectionMethod.Transition, CorrectionMethod.PE}:
            batch.fwd_equivalent_idxs = torch.tensor(
                sum(
                    [idxs for traj in trajectories for idxs in traj.fwd_equivalents], []
                )
            )
            batch.fwd_equivalent_batch = torch.repeat_interleave(
                torch.tensor(
                    [
                        len(idxs)
                        for traj in trajectories
                        for idxs in traj.fwd_equivalents
                    ]
                )
            )
        if self.correction_method == CorrectionMethod.RewardScaling:
            batch.last_log_sym = torch.tensor(
                [traj.last_log_sym for traj in trajectories], dtype=torch.float
            )
        if self.unif_backward:
            batch.traj_bck_log_probs = torch.tensor(
                [sum(traj.bck_log_probs) for traj in trajectories], dtype=torch.float
            )
        elif self.correction_method in {CorrectionMethod.Transition, CorrectionMethod.PE}:
            batch.bck_equivalent_idxs = torch.tensor(
                sum(
                    [idxs for traj in trajectories for idxs in traj.bck_equivalents], []
                )
            )
            batch.bck_equivalent_batch = torch.repeat_interleave(
                torch.tensor(
                    [
                        len(idxs)
                        for traj in trajectories
                        for idxs in traj.bck_equivalents
                    ]
                )
            )
        return batch

    def make_transition_batch(self, transitions: list[Transition]):
        batch = self.env.collate(
            [tran.torch_graphs[0] for tran in transitions],
            set_backward=not self.unif_backward,
        )
        batch.next_batch = self.env.collate(
            [tran.torch_graphs[1] for tran in transitions],
            set_backward=not self.unif_backward,
        )
        batch.log_rewards = torch.tensor(
            [tran.log_reward for tran in transitions], dtype=torch.float
        )
        batch.fwd_action_idxs = torch.tensor(
            [tran.fwd_action_idx for tran in transitions], dtype=torch.long
        )
        batch.bck_action_idxs = torch.tensor(
            [tran.bck_action_idx for tran in transitions], dtype=torch.long
        )
        if self.correction_method in {CorrectionMethod.Transition, CorrectionMethod.PE}:
            batch.fwd_equivalent_idxs = torch.tensor(
                sum([tran.fwd_equivalents for tran in transitions], [])
            )
            batch.fwd_equivalent_batch = torch.repeat_interleave(
                torch.tensor([len(tran.fwd_equivalents) for tran in transitions])
            )
        if self.correction_method == CorrectionMethod.RewardScaling:
            batch.last_log_sym = torch.tensor(
                [tran.last_log_sym for tran in transitions], dtype=torch.float
            )
        elif self.correction_method == CorrectionMethod.FlowScaling:
            batch.log_sym_ratio = torch.tensor(
                [tran.log_sym_ratio for tran in transitions], dtype=torch.float
            )
        if self.unif_backward:
            batch.bck_log_probs = torch.tensor(
                [tran.bck_log_prob for tran in transitions], dtype=torch.float
            )
        elif self.correction_method in {CorrectionMethod.Transition, CorrectionMethod.PE}:
            batch.bck_equivalent_idxs = torch.tensor(
                sum([tran.bck_equivalents for tran in transitions], [])
            )
            batch.bck_equivalent_batch = torch.repeat_interleave(
                torch.tensor([len(tran.bck_equivalents) for tran in transitions])
            )
        batch.is_terminal = torch.tensor(
            [tran.is_terminal for tran in transitions], dtype=torch.bool
        )
        return batch

    @torch.no_grad()
    def sample_trajectories(
        self, num_samples: int, epsilon: float = 0.0
    ) -> list[Trajectory]:
        dones = [False] * num_samples
        states = [self.env.new() for _ in range(num_samples)]
        trajectories = [Trajectory() for _ in range(num_samples)]

        while states:
            states_info = self.env.featurize(
                states, set_backward=not self.unif_backward
            )
            torch_graphs, children_lst = states_info["graph"], states_info["children"]
            if epsilon < random.random():
                actions_idxs = [
                    random.randint(0, len(ch.states) - 1) for ch in children_lst
                ]
            else:
                batch = self.env.collate(torch_graphs)
                fwd_dist, _, _ = self.model(batch)
                actions_idxs = fwd_dist.sample().tolist()

            graph_actions = [
                children_lst[i].actions[idx] for i, idx in enumerate(actions_idxs)
            ]
            not_dones = [i for i, done in enumerate(dones) if not done]
            next_states = []

            for i, j in zip(not_dones, range(len(states))):
                traj = trajectories[i]
                traj.states.append(states[j])
                traj.torch_graphs.append(torch_graphs[j])
                traj.actions.append(graph_actions[j])
                traj.fwd_action_idxs.append(actions_idxs[j])
                if self.correction_method == CorrectionMethod.Transition:
                    equivalent_idxs = self.env.equivalent_action_idxs(
                        children_lst[j], actions_idxs[j]
                    )
                    traj.fwd_equivalents.append(equivalent_idxs)
                elif self.correction_method == CorrectionMethod.PE:
                    pe = random_walk_pe_with_node_types(torch_graphs[j], self.pe_length)
                    equivalent_idxs = self.env.equivalent_action_idxs_using_pe(
                        children_lst[j], actions_idxs[j], pe
                    )
                    traj.fwd_equivalents.append(equivalent_idxs)

                if graph_actions[j].type.name == "Stop":
                    dones[i] = True
                else:
                    c, idx = children_lst[j].states, actions_idxs[j]
                    next_states.append(c[idx])
            states = next_states
        return trajectories


class GFNTrainer:
    def __init__(
        self,
        model: PolicyNet,
        env: GraphEnv,
        logger: logging.Logger = None,
        unif_backward=True,
        correction_method=None,
        training_method="tb",
        sampling_tau=0.99,
        grad_norm_clip=1.0,
        lr=0.001,
        min_lr=None,
        exploration_epsilon=0.1,
        train_size=64,
        sampling_size=32,
        sampling_freq=4,
        buffer_size=100,
        num_workers=0,
    ):
        assert correction_method.lower() in {
            "vanilla",
            "transition",
            "pe",
            "flowscaling",
            "rewardscaling",
        }
        assert training_method in {"tb", "db"}
        self.model = model
        self.env = env
        self.unif_backward = unif_backward
        self.correction_method = correction_method
        self.training_method = training_method
        self.sampling_tau = sampling_tau
        self.grad_norm_clip = grad_norm_clip

        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger

        self.target_model = None
        if not self.unif_backward:
            self.model.add_backward_policy()
        if training_method == "tb":
            self.model.add_logZ()
            proxy = ModelProxy(model, num_workers)
        elif training_method == "db":
            self.model.add_state_flow()
            self.sampling_model = deepcopy(self.model)
            proxy = ModelProxy(self.model, num_workers)
        else:
            raise ValueError()

        self.optimizer = self.config_optim(lr)
        self.min_lr = min_lr
        self.scheduler = None
        if min_lr is not None:
            min_mul = self.min_lr / lr
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                        lr_lambda=lambda n_iter: max(min_mul, 0.995 ** n_iter))
        self.correction_method = {
            CorrectionMethod(i).name.lower():CorrectionMethod(i) for i in range(1, 6)
        }[correction_method.lower()]
        
        dataset = GFNDataset(
            proxy.placeholder,
            env,
            transition=(training_method == "db"),
            train_size=train_size,
            sampling_size=sampling_size,
            sampling_freq=sampling_freq,
            buffer_size=buffer_size,
            exploration_epsilon=exploration_epsilon,
            unif_backward=unif_backward,
            correction_method=self.correction_method,
        )
        self.dataloader = DataLoader(
            dataset,
            worker_init_fn=proxy.worker_init_fn,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 1 else None,
            persistent_workers=bool(num_workers > 1),
            batch_size=None,
        )
        self.callbacks = defaultdict(list)

        #                       <     iter_dt     >
        # |-------------|-------|-----------------|------------>
        # ^iter_start                             ^iter_time
        self.iter_num = 0
        self.iter_start = 0.0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.break_flg = False  # stop training if True

    def config_optim(self, lr):
        logZ_params = [
            param for name, param in self.model.named_parameters() if "logZ" == name
        ]
        other_params = [
            param for name, param in self.model.named_parameters() if "logZ" != name
        ]
        optimizer = torch.optim.Adam(other_params, lr=lr)
        optimizer.add_param_group({"params": logZ_params, "lr": lr * 100})
        return optimizer

    def add_callback(self, on_event: str, callback):
        self.callbacks[on_event].append(callback)

    def set_callback(self, on_event: str, callback):
        self.callbacks[on_event] = [callback]

    def trigger_callbacks(self, on_event: str):
        for callback in self.callbacks.get(on_event, []):
            callback(self)

    def tb_loss(self, batch):
        fwd_dist, bck_dist, _ = self.model.forward(
            batch, return_backward_dist=not self.unif_backward
        )

        # compute forward flow
        if self.correction_method in {CorrectionMethod.Transition, CorrectionMethod.PE}:
            fwd_log_probs = fwd_dist.subset_logsumexp(
                batch.fwd_equivalent_idxs, batch.fwd_equivalent_batch
            )
        else:
            fwd_log_probs = fwd_dist.log_prob(batch.fwd_action_idxs)
        traj_fwd_log_probs = scatter_add(
            fwd_log_probs, batch.traj_fwd_batch, dim_size=batch.num_trajectories
        )
        fwd_log_flow = self.model.logZ + traj_fwd_log_probs

        # compute backward flow
        if self.unif_backward:
            traj_bck_log_probs = batch.traj_bck_log_probs
        else:
            if self.correction_method in {CorrectionMethod.Transition, CorrectionMethod.PE}:
                bck_log_probs = bck_dist.subset_logsumexp(
                    batch.bck_equivalent_idxs, batch.bck_equivalent_batch
                )
            else:
                bck_log_probs = bck_dist.log_prob(batch.bck_action_idxs)
            traj_bck_log_probs = scatter_add(
                bck_log_probs, batch.traj_bck_batch, dim_size=batch.num_trajectories
            )

        if self.correction_method == CorrectionMethod.RewardScaling:
            log_rewards = (
                batch.log_rewards + batch.last_log_sym - self.env.first_log_sym
            )
        else:
            log_rewards = batch.log_rewards

        # compute loss
        bck_log_flow = log_rewards + traj_bck_log_probs
        loss = (fwd_log_flow - bck_log_flow).square().mean()
        return loss

    def db_loss(self, batch):
        fwd_dist, bck_dist, fwd_log_flow = self.model.forward(
            batch, 
            return_backward_dist=not self.unif_backward, 
            return_flow=True
        )

        # compute forward flow
        if self.correction_method in {CorrectionMethod.Transition, CorrectionMethod.PE}:
            fwd_log_probs = fwd_dist.subset_logsumexp(
                batch.fwd_equivalent_idxs, batch.fwd_equivalent_batch
            )
        else:
            fwd_log_probs = fwd_dist.log_prob(batch.fwd_action_idxs)

        fwd_log_outflow = fwd_log_flow + fwd_log_probs

        # compute backward flow
        with torch.no_grad():
            _, _, bck_log_flow = self.sampling_model(
                batch.next_batch,
                return_backward_dist=not self.unif_backward,
                return_flow=True,
            )
            bck_log_flow = torch.where(batch.is_terminal, 0.0, bck_log_flow.detach())
            if self.correction_method == CorrectionMethod.RewardScaling:
                log_rewards = batch.log_rewards + batch.last_log_sym
            elif self.correction_method == CorrectionMethod.FlowScaling:
                log_rewards = batch.log_rewards + batch.log_sym_ratio
            else:
                log_rewards = batch.log_rewards

            if self.unif_backward:
                bck_log_probs = batch.bck_log_probs
            else:
                if self.correction_method in {CorrectionMethod.Transition, CorrectionMethod.PE}:
                    bck_log_probs = bck_dist.subset_logsumexp(
                        batch.bck_equivalent_idxs, batch.bck_equivalent_batch
                    )
                else:
                    bck_log_probs = bck_dist.log_prob(batch.bck_action_idxs)
            bck_log_inflow = bck_log_flow + bck_log_probs + log_rewards

        # compute loss
        loss = (fwd_log_outflow - bck_log_inflow).square().mean()
        return loss

    def run(self, iters=1, device="cpu", print_every=1):
        max_iters = self.iter_num + iters
        self.model.train()
        self.model.to(device)
        if self.sampling_model:
            self.sampling_model.to(device)

        self.trigger_callbacks("on_train_start")
        self.iter_time = self.iter_start = time.time()

        for batch in self.dataloader:
            if self.training_method == "tb":
                loss = self.tb_loss(batch)
            elif self.training_method == "db":
                loss = self.db_loss(batch)
            else:
                raise ValueError()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_norm_clip
            )
            self.optimizer.step()
            self.model.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()

            if self.sampling_model:
                self.update_sampling_model()

            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            if self.iter_num % print_every == 0:
                logs = {
                    "loss": loss.item(),
                    "grad_norm": grad_norm.item(),
                    "mean_log_reward": batch.log_rewards.mean().item(),
                    "iteration": self.iter_num,
                    "current_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "train_time": self.iter_dt,
                    "sampling_time": batch.sampling_time
                }
                self.logger.info(str(logs))

            self.trigger_callbacks("on_batch_end")

            if self.iter_num >= max_iters or self.break_flg:
                self.trigger_callbacks("on_train_end")
                break

    def save(self, path):
        torch.save(
            {
                "iteration": self.iter_num,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "target_model_state_dict": (
                    self.target_model.state_dict() if self.target_model else None
                ),
            },
            path,
        )

    def load(self, path, map_location="cpu"):
        state_dicts = torch.load(path, map_location=map_location)
        self.iter_num = state_dicts["iteration"]
        self.model.load_state_dict(state_dicts["model_state_dict"])
        if "optimizer_state_dict" in state_dicts:
            self.optimizer.load_state_dict(state_dicts["optimizer_state_dict"])
        if self.target_model:
            self.target_model.load_state_dict(
                state_dicts["target_model_state_dict"]
            )

    def update_target_model(self):
        for model_param, sampling_param in zip(
            self.model.parameters(), self.target_model.parameters()
        ):
            sampling_param.data.mul_(self.sampling_tau).add_(
                model_param.data * (1 - self.sampling_tau)
            )
