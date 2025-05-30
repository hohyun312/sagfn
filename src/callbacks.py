import os
from scipy.stats import spearmanr, linregress
from scipy.spatial.distance import jensenshannon
import numpy as np

import torch

from src.graphenv import GraphEnv
from src.models import PolicyNet


@torch.no_grad()
def compute_terminating_logprobs(env: GraphEnv, model: PolicyNet):
    state = env.new()
    is_training = model.training
    model.eval()
    state.hash = env.hash(state)
    frontier = [state]
    incoming_logprobs = {state.hash: 0.0}
    terminating_logprobs = {}

    def init_frontier(states):
        info = env.featurize(states)
        batch = env.collate(info["graph"])
        dist, _, _ = model(batch)
        children_sizes = [len(children.states) for children in info["children"]]
        logprobs_lst = dist.log_prob().split(children_sizes)
        for state, children, logprobs in zip(states, info["children"], logprobs_lst):
            state.next_states = children.states
            state.actions = children.actions
            state.action_logprobs = logprobs.tolist()

    while frontier:
        init_frontier(frontier)
        next_frontier = []
        for state in frontier:
            for child, action, logprob in zip(
                state.next_states, state.actions, state.action_logprobs
            ):
                if action.type.name == "Stop":
                    terminating_logprobs[state.hash] = (
                        incoming_logprobs[state.hash] + logprob
                    )
                else:
                    child.hash = env.hash(child)
                    if child.hash not in incoming_logprobs:
                        incoming_logprobs[child.hash] = (
                            incoming_logprobs[state.hash] + logprob
                        )
                        next_frontier.append(child)
                    else:
                        incoming_logprobs[child.hash] = np.logaddexp(
                            incoming_logprobs[child.hash],
                            incoming_logprobs[state.hash] + logprob,
                        )
        frontier = next_frontier
    model.train(is_training)
    return terminating_logprobs


def get_exact_eval_callback(env: GraphEnv, model: PolicyNet, eval_every=20):
    state_info = env.terminal_states_info()
    all_rewards = np.array(
        [
            state_info[x]["reward"]
            for x in sorted(state_info)
            if state_info[x]["reward"] is not None
        ]
    )
    target_probs = all_rewards / np.sum(all_rewards)

    def callback(trainer):
        is_training = model.training
        model.eval()
        if trainer.iter_num % eval_every == 0:
            terminating_logprobs = compute_terminating_logprobs(env, model)
            model_probs = np.exp(
                [terminating_logprobs[x] for x in sorted(terminating_logprobs)]
            )
            model_probs = model_probs / model_probs.sum()
            try:
                spearman = spearmanr(target_probs, model_probs).statistic
                pearson = linregress(target_probs, model_probs).rvalue
            except:
                spearman, pearson = None, None
                pass
            l1 = np.linalg.norm(target_probs - model_probs, ord=1)
            l2 = np.linalg.norm(target_probs - model_probs, ord=2)
            linf = abs(np.max(target_probs - model_probs))
            js = jensenshannon(target_probs, model_probs)
            logs = {
                "iteration": trainer.iter_num,
                "spearman": spearman,
                "pearson": pearson,
                "l1": l1,
                "l2": l2,
                "sup": linf,
                "js": js,
            }
            trainer.logger.info(str(logs))
        model.train(is_training)

    return callback


def get_model_save_callback(save_dir, prefix="model", save_every=20):
    def callback(trainer):
        if trainer.iter_num % save_every == 0:
            path = os.path.join(save_dir, f"{prefix}-{trainer.iter_num}.pt")
            torch.save(trainer.model.state_dict(), path)

    return callback


def get_checkpoint_callback(save_dir, name="ckpt", save_every=20):
    def callback(trainer):
        if trainer.iter_num % save_every == 0:
            path = os.path.join(save_dir, f"{name}.pt")
            trainer.save(path)

    return callback


def get_break_callback(save_dir, name="ckpt", time_limit=10 * 60):
    def callback(trainer):
        traintime_min = (trainer.iter_time - trainer.iter_start) / 60
        if traintime_min >= time_limit:
            if save_dir:
                path = os.path.join(save_dir, f"{name}.pt")
                trainer.save(path)
            trainer.break_flg = True

    return callback
