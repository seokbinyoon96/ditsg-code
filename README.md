# DiTSG: Context-Aware Air Traffic Scene Generation with Multi-Agent Diffusion Transformers

[Project Page](https://seokbinyoon96.github.io/ditsg/) | [arXiv](https://arxiv.org/abs/2610.16959)

## Overview

![DiTSG architecture: multi-agent diffusion transformer and sampling loop.](https://seokbinyoon96.github.io/ditsg/static/images/architecture.png)

**Diffusion Transformers for Traffic Scene Generation (DiTSG)** is a context-aware generative model that learns from recorded air traffic operations to jointly generate multiple aircraft trajectories. Its denoising backbone uses stacked multi-agent inverted Transformer (MAIFormer) layers to model dependencies between aircraft. Conditioned on each aircraft's required time of arrival (RTA), DiTSG progressively refines noise into diverse traffic scenes that preserve realistic inter-aircraft interactions.

## Structure

```
config.py        hyperparameters
embedding.py     sinusoidal time embedding (t, tau)
block.py         MAI block (masked multivariate attn -> agent attn -> FFN, adaLN-Zero)
model.py         DiTSG
train.py         flow matching loss, EMA, training loop
generate.py      Euler ODE sampling
```

## Train

```python
from model import build_model
from train import train

# loader yields (traj [B, T, N*3], tau [B, N]), normalized, same N per batch
model = build_model(seq_len=20)
train(model, loader)
```

## Generate

```bash
python generate.py --ckpt result/checkpoints/ditsg_best.pt --tau 0.1 0.3 0.5 --n 10
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{yoon2026ditsg,
  title={DiTSG: Context-Aware Air Traffic Scene Generation with Multi-Agent Diffusion Transformers},
  author={Yoon, Seokbin and Kim, Kyungmin and Kwon, Taejung and Lee, Keumjin and Wei, Peng},
  journal={arXiv preprint arXiv:2610.16959},
  year={2026}
}
```
