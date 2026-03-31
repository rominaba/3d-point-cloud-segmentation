import torch
import numpy as np
import random
import gc
from typing import Optional
import logging

DATE_FORMAT = "%Y-%m-%d-%H:%M"
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt=DATE_FORMAT,
            )
        )
        logger.addHandler(handler)
    return logger

def choose_device()->str:
    if torch.backends.cuda.is_built():
        # usually on Windows machines with GPU
        device = "cuda"
    elif torch.backends.mps.is_built():
        # usually on MAC
        device = "mps"
    else:
        # if not we should use our CPU
        device = "cpu"
    get_logger("utils").info(f"Chosen device: {device}")
    return device


def collect_garbage(device: str) -> None:
    """GPU garbage collection
    """
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

def set_seed(seed: Optional[int] = None) -> None:
  """Set seed for reproducibility
  """
  seed  = 1234 if seed is None else seed
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

