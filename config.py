"""Hyperparameters and paths."""

from pathlib import Path

import torch

RESULT_DIR = Path(__file__).parent / "result"
RUN_NAME = "ditsg"           # -> <RUN_NAME>_best.pt / <RUN_NAME>_last.pt

# model
NF = 3                       # lat, lon, alt
D_MODEL = 128
HEAD = 4
D_FF = 512
NUM_LAYERS = 6
DROPOUT = 0.01
TIME_EMB_SCALE = 1000.0      # t, tau in [0, 1] -> scale before sinusoidal

# training
EPOCHS = 300
LR = 2e-4
WEIGHT_DECAY = 1e-4
SEP_WEIGHT = 1e-4            # separation penalty (0: off)
EMA_DECAY = 0.999            # 0: off
PATIENCE = 0                 # early stopping (0: off)
SEED = 0

# flow matching
T_SAMPLING = "logitnormal"   # logitnormal / uniform
T_LOGIT_MEAN = 0.0
T_LOGIT_STD = 1.0
T_EPS = 1e-3                 # clamp on (1 - t)
SAMPLE_STEPS = 50            # Euler steps

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

HPARAMS = ("RUN_NAME", "NF", "D_MODEL", "HEAD", "D_FF", "NUM_LAYERS", "DROPOUT",
           "TIME_EMB_SCALE", "EPOCHS", "LR", "WEIGHT_DECAY", "SEP_WEIGHT", "EMA_DECAY",
           "PATIENCE", "SEED", "T_SAMPLING", "T_LOGIT_MEAN", "T_LOGIT_STD", "T_EPS",
           "SAMPLE_STEPS")


def _derive():
    g = globals()
    g["CHECKPOINT_DIR"] = Path(RESULT_DIR) / "checkpoints"
    g["BEST_PATH"] = g["CHECKPOINT_DIR"] / f"{RUN_NAME}_best.pt"
    g["LAST_PATH"] = g["CHECKPOINT_DIR"] / f"{RUN_NAME}_last.pt"


def update(**kw):
    """config.update(LR=1e-4, RUN_NAME="x"); unknown keys are ignored."""
    g = globals()
    for k, v in kw.items():
        k = k.upper()
        if k in HPARAMS or k == "RESULT_DIR":
            g[k] = v
    _derive()


def hparams():
    g = globals()
    return {k: g[k] for k in HPARAMS}


_derive()
