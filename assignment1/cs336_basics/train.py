import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import time
from typing import Optional
import mmap

from cs336_basics.BPETokenizer import BPETokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.Loss import cross_entropy, perplexity
from cs336_basics.optimizer import get_adamw_cls, get_lr_cosine_schedule,Adamw
from cs336_basics.gradient_clip import gradient_clipping
from cs336_basics.check_point import save_checkpoint, load_checkpoint
from cs336_basics.data import get_batch
import wandb
import argparse
import json
import yaml
import math

