from omegaconf import OmegaConf

import torch

from src.graphenv import GraphEnv, IllustrativeEnv, MolEnv
from src.trainer import GFNTrainer
from src.models import PolicyNet
from src.utils import get_logger, set_main_process_seed
from src.callbacks import (
    get_exact_eval_callback,
    get_model_save_callback,
    get_break_callback,
)


if __name__ == "__main__":
    import sys, os
    
    if len(sys.argv) == 4:

        conf = OmegaConf.load(sys.argv[1])

        # set seed
        seed = int(sys.argv[2])
        set_main_process_seed(seed)

        # save_dir: directory to store checkpoints and log files.
        save_dir = os.path.join(sys.argv[3], conf.directory, f"{conf.name}-{seed}")

        os.makedirs(save_dir, exist_ok=True)
        print("Make directory at:", save_dir)

        # save config file at save_dir
        conf_path = os.path.join(save_dir, "config.yaml")
        with open(conf_path, "w") as f:
            OmegaConf.save(conf, f)

        print("Save config file at:", conf_path)
        print(conf)

        # set device
        if conf.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(conf.device)
        print(device)

        # logging file
        logging_path = os.path.join(save_dir, "logs.txt")
        logger = get_logger(path=logging_path)

        # set environment
        if conf.env.name == "graph":
            env = GraphEnv(**conf.env)
        elif conf.env.name == "illustrative":
            env = IllustrativeEnv(**conf.env)
        elif conf.env.name == "mol":
            env = MolEnv(**conf.env)
        else:
            raise ValueError(f"{conf.env.name} is not supported")

        model = PolicyNet(env, **conf.model)
        trainer = GFNTrainer(model, env, logger, **conf.trainer)

        if conf.eval_every:
            eval_model = trainer.sampling_model if trainer.sampling_model else model
            trainer.add_callback(
                "on_train_start", get_exact_eval_callback(env, eval_model, eval_every=1)
            )
            trainer.add_callback(
                "on_batch_end",
                get_exact_eval_callback(env, eval_model, eval_every=conf.eval_every),
            )

        # set callbacks
        if conf.save_every:
            trainer.add_callback(
                "on_batch_end",
                get_model_save_callback(save_dir, "model", save_every=conf.save_every),
            )

        trainer.add_callback(
            "on_batch_end", get_break_callback(save_dir, name="ckpt", time_limit=712)
        )

        ckpt_path = os.path.join(save_dir, "final.pt")
        max_iters = conf.num_iters
        if os.path.exists(ckpt_path):
            print("Load checkpoint:", ckpt_path)
            trainer.load(ckpt_path)
            max_iters = max_iters - trainer.iter_num
        
        print("Train start")
        trainer.run(iters=max_iters)
        trainer.save(ckpt_path)
        print('Save:', ckpt_path)
    else:
        print("Usage: python main.py <config_path> <seed> <save_dir>")
        sys.exit(1)