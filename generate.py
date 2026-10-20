"""Generate scenes with a trained DiTSG (Euler ODE, noise t=0 -> data t=1).

    python generate.py --ckpt result/checkpoints/ditsg_best.pt --tau 0.1 0.3 0.5 --n 10

tau and the output trajectories are in the normalized scale used for training.
"""

import argparse

import numpy as np
import torch

import config
from model import build_model


def load_model(path, key="ema"):
    """Build DiTSG from the checkpoint's config and load its weights (EMA by default)."""
    ck = torch.load(path, map_location=config.DEVICE, weights_only=False)
    config.update(**ck.get("config", {}))
    state = ck.get(key, ck["model"])
    model = build_model(seq_len=state["embedding.weight"].shape[1], verbose=False)
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def sample(model, tau, steps=None, seed=None):
    """tau [n, N] -> x [n, T, N*F]"""
    steps = config.SAMPLE_STEPS if steps is None else steps
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    tau = torch.as_tensor(tau, dtype=torch.float32, device=config.DEVICE)
    n, num_agents = tau.shape

    z = torch.randn(n, model.seq_len, num_agents * config.NF, device=config.DEVICE)
    ts = torch.linspace(0.0, 1.0, steps + 1, device=config.DEVICE)
    for i in range(steps):
        t, t_next = ts[i], ts[i + 1]
        x_pred = model(z, t.expand(n), tau)
        v_pred = (x_pred - z) / (1.0 - t).clamp(min=config.T_EPS)
        z = z + (t_next - t) * v_pred
    return z


@torch.no_grad()
def generate(model, tau, steps=None, seed=0, chunk=512):
    """Chunked sample(); tau [n, N] -> x [n, T, N*F] on CPU."""
    tau = torch.as_tensor(tau, dtype=torch.float32)
    outs = []
    for start in range(0, len(tau), chunk):
        outs.append(sample(model, tau[start:start + chunk], steps, seed + start).cpu())
    return torch.cat(outs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(config.BEST_PATH))
    p.add_argument("--tau", type=float, nargs="+", required=True, help="normalized, one per agent")
    p.add_argument("--n", type=int, default=10, help="scenes per tau")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="generated.npz")
    args = p.parse_args()

    model = load_model(args.ckpt)
    tau = torch.tensor(args.tau).repeat(args.n, 1)        # [n, N]
    x = generate(model, tau, args.steps, args.seed)
    np.savez(args.out, traj=x.numpy(), tau=tau.numpy())
    print(f"saved {args.out}: traj {tuple(x.shape)}")


if __name__ == "__main__":
    main()
