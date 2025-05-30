'''
Logging & multiprocessing modules adapted from https://github.com/recursionpharma/gflownet/tree/trunk.
'''
import logging
import random
import numpy as np
import torch
import queue
import threading
import pickle
import numpy as np

import torch
import torch.multiprocessing as mp
from torch.utils.data import get_worker_info


def get_logger(name=None, path=None, level="DEBUG"):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if path is None:
        logger.addHandler(logging.StreamHandler())
    else:
        logger.addHandler(logging.FileHandler(path, mode="a"))
    return logger


_main_process_seed = [0]


def set_main_process_seed(seed):
    set_seed(seed)
    _main_process_seed[0] = seed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ModelPlaceholder:
    def __init__(self, in_queue, out_queue, pickle_messages):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.pickle_messages = pickle_messages

    def encode(self, m):
        if self.pickle_messages:
            return pickle.dumps(m)
        return m

    def decode(self, m):
        if self.pickle_messages:
            m = pickle.loads(m)
        if isinstance(m, Exception):
            print("Received exception from main process, reraising.")
            raise m
        return m

    def __call__(self, batch):
        self.in_queue.put(self.encode(batch))
        return self.decode(self.out_queue.get())

    def __getattr__(self, name):
        def method_wrapper(*a, **kw):
            self.in_queue.put(self.encode((name, a, kw)))
            return self.decode(self.out_queue.get())

        return method_wrapper


class ModelProxy:
    def __init__(self, model, num_workers, pickle_messages=False):
        self.model = model
        self.num_workers = num_workers
        self.pickle_messages = pickle_messages
        if num_workers > 1:
            self.in_queues = [mp.Queue() for _ in range(num_workers)]
            self.out_queues = [mp.Queue() for _ in range(num_workers)]
            self.placeholder = ModelPlaceholder(self.in_queues[0], self.out_queues[0], pickle_messages)
            self.stop = threading.Event()
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
        else:
            self.placeholder = model

    def encode(self, m):
        if self.pickle_messages:
            return pickle.dumps(m)
        return m

    def decode(self, m):
        if self.pickle_messages:
            return pickle.loads(m)
        return m

    def worker_init_fn(self, worker_id):
        if self.num_workers > 1:
            worker_info = get_worker_info()
            model = worker_info.dataset.model
            model.in_queue = self.in_queues[worker_info.id]
            model.out_queue = self.out_queues[worker_info.id]
            set_seed(_main_process_seed[0] + worker_info.id + 42)
        else:
            pass

    def run(self):
        while not self.stop.is_set():
            for in_queue, out_queue in zip(self.in_queues, self.out_queues):
                try:
                    batch = self.decode(in_queue.get(True, 1e-5))
                except queue.Empty:
                    continue

                with torch.no_grad():
                    fwd_dist, bck_dist, values = self.model(batch.to(self.model.device))
                    output = (
                        fwd_dist.to("cpu"),
                        bck_dist.to("cpu") if bck_dist is not None else None,
                        values.to("cpu") if values is not None else None,
                    )

                out_queue.put(self.encode(output))

    def stop(self):
        self.stop.set()

