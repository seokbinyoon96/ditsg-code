"""Train DiTSG with conditional flow matching (x-prediction).

    t      ~ logit-normal
    z      = t * x + (1 - t) * e,   e ~ N(0, I)
    v      = (x - z) / (1 - t)
    v_pred = (net(z, t, tau) - z) / (1 - t)
    loss   = ||v - v_pred||^2 + sep_weight * separation
"""

import copy

import torch
from tqdm.auto import tqdm

import config

NF = config.NF
SEP_H_NM = 3.0
SEP_V_FT = 1000.0


def sample_t(n, device):
    """flow time [n] in (0, 1)"""
    if config.T_SAMPLING == "uniform":
        t = torch.rand(n, device=device)
    else:
        t = torch.sigmoid(torch.randn(n, device=device) * config.T_LOGIT_STD + config.T_LOGIT_MEAN)
    return t.clamp(max=1.0 - config.T_EPS)


def separation_penalty(x, scalers, t=-1):
    """Pairwise 3 NM / 1000 ft violation at time step t. x normalized [B, T, N*F]."""
    N = x.shape[2] // NF
    if N < 2:
        return x.sum() * 0.0
    p = x[:, t, :]
    i, j = torch.triu_indices(N, N, offset=1, device=x.device)
    dlat = (p[:, 0::NF][:, i] - p[:, 0::NF][:, j]) * scalers.lat2nm
    dlon = (p[:, 1::NF][:, i] - p[:, 1::NF][:, j]) * scalers.lon2nm
    dh = torch.sqrt(dlat ** 2 + dlon ** 2 + 1e-8)
    dv = (p[:, 2::NF][:, i] - p[:, 2::NF][:, j]).abs() * scalers.alt2ft
    h = torch.clamp(1.0 - dh / SEP_H_NM, min=0.0)
    v = torch.clamp(1.0 - dv / SEP_V_FT, min=0.0)
    return ((h * v) ** 2).mean()


def flow_loss(model, x, tau, scalers=None, sep_weight=0.0):
    """x [B, T, N*F], tau [B, N] (normalized)"""
    B, num_agents = x.shape[0], x.shape[2] // NF

    t = sample_t(B, x.device)
    tb = t.view(-1, 1, 1)
    one_minus_t = (1.0 - tb).clamp(min=config.T_EPS)

    e = torch.randn_like(x)
    z = tb * x + (1.0 - tb) * e
    v = (x - z) / one_minus_t

    x_pred = model(z, t, tau)
    v_pred = (x_pred - z) / one_minus_t

    # sum over (T, F), mean over agents and scenes
    matching = (v - v_pred).pow(2).sum(dim=(1, 2)).mean() / num_agents

    if sep_weight > 0:
        separation = separation_penalty(x_pred, scalers)
    else:
        separation = x_pred.sum() * 0.0

    loss = matching + sep_weight * separation
    return loss, {"matching": matching.item(), "separation": separation.item()}


class EMA:
    """EMA copy of the weights, used for sampling."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p.detach(), 1.0 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


def train_epoch(model, loader, optimizer, scalers=None, ema=None, epoch=0):
    model.train()
    tot = {"loss": 0.0, "matching": 0.0, "sep": 0.0}
    n = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False):
        traj = batch[0].to(config.DEVICE)
        tau = batch[1].to(config.DEVICE)

        loss, logs = flow_loss(model, traj, tau, scalers, config.SEP_WEIGHT)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)

        b = traj.size(0)
        n += b
        tot["loss"] += loss.item() * b
        tot["matching"] += logs["matching"] * b
        tot["sep"] += logs["separation"] * b

    return {k: v / max(n, 1) for k, v in tot.items()}


def train(model, loader, scalers=None):
    """
    loader  : yields (traj [B, T, N*F], tau [B, N], ...), normalized, same N per batch
    scalers : object with lat2nm / lon2nm / alt2ft (only for SEP_WEIGHT > 0)
    """
    torch.manual_seed(config.SEED)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    ema = EMA(model, config.EMA_DECAY) if config.EMA_DECAY > 0 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR,
                                  weight_decay=config.WEIGHT_DECAY)
    best_loss, best_epoch, history = float("inf"), 0, []

    for epoch in range(1, config.EPOCHS + 1):
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)
        tr = train_epoch(model, loader, optimizer, scalers, ema, epoch)
        history.append({"epoch": epoch, "train": tr})

        state = {"model": model.state_dict(), "epoch": epoch, "config": config.hparams()}
        if ema is not None:
            state["ema"] = ema.shadow.state_dict()

        is_best = tr["loss"] < best_loss
        if is_best:
            best_loss, best_epoch = tr["loss"], epoch
            torch.save({**state, "train_loss": tr["loss"]}, config.BEST_PATH)
        torch.save({**state, "optimizer": optimizer.state_dict(), "history": history},
                   config.LAST_PATH)

        print(f"[epoch {epoch:03d}/{config.EPOCHS}] loss {tr['loss']:.6f} "
              f"(matching {tr['matching']:.6f}, sep {tr['sep']:.6f})"
              + ("  <- best" if is_best else ""))

        if config.PATIENCE and epoch - best_epoch >= config.PATIENCE:
            print(f"early stop @ epoch {epoch}")
            break

    print(f"best loss {best_loss:.6f} @ epoch {best_epoch}")
    return model, ema, history
