# DiTSG: Context-Aware Air Traffic Scene Generation with Multi-Agent Diffusion Transformers

[Project Page](https://seokbinyoon96.github.io/ditsg/) | [arXiv](https://arxiv.org/abs/2610.16959)

## Overview

![DiTSG architecture: multi-agent diffusion transformer and sampling loop.](https://seokbinyoon96.github.io/ditsg/static/images/architecture.png)

We propose Diffusion Transformers for Traffic Scene Generation (DiTSG), a context-aware generative model for multi-agent air traffic scenes, where all aircraft trajectories are generated jointly and each trajectory is modeled in the context of the others. DiTSG employs stacked multi-agent inverted Transformer (MAIFormer) layers as the denoising backbone of a diffusion transformer. Generation is further conditioned on agent-specific required times of arrival (RTAs), enabling different temporal traffic configurations. Starting from noise, DiTSG iteratively refines the trajectories of all aircraft in a scene while modeling inter-aircraft dependencies at every denoising step. This joint denoising process enables the generated trajectories to remain mutually consistent while preserving realistic traffic interactions.


## Structure

```
config.py        hyperparameters
embedding.py     sinusoidal time embedding (t, tau)
block.py         MAI block (masked multivariate attn -> agent attn -> FFN, adaLN-Zero)
model.py         DiTSG
train.py         flow matching loss, EMA, training loop
generate.py      Euler ODE sampling
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
