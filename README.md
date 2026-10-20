# DiTSG: Context-Aware Air Traffic Scene Generation with Multi-Agent Diffusion Transformers

Project page: https://seokbinyoon96.github.io/ditsg/

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
