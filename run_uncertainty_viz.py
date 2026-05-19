#!/usr/bin/env python3
"""
run_uncertainty_viz.py  —  Live uncertainty visualization for SystemDynamicsEnsemble.

Standalone script: reconstructs the model directly from the checkpoint state_dict
(no Isaac Lab / rsl_rl installation required — only torch, numpy, matplotlib).

Architecture (reverse-engineered from checkpoint):
  - Shared GRU backbone   state_base.memory.rnn        input=57 (45+12), hidden=256, layers=2
  - E independent heads   state_heads.{0..E-1}         GRU-out(256) → MLP → (mean 45, logstd 45)
  - Auxiliary backbone    auxiliary_base.memory.rnn     same GRU structure (contact / termination)
  - E auxiliary heads     auxiliary_heads.{0..E-1}      → contact(8), termination(1)

Usage:
    python run_uncertainty_viz.py \
        --checkpoint checkpoints_paws/model_2000_pretrain.pt \
        --data       utils_paws/walk_spot2.npz \
        [--device cpu] [--pause 0.05] [--update_every 10]
"""

from __future__ import annotations
import argparse, sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Go2 normalization constants (from Go2FlatConfig) ─────────────────────────
_STATE_MEAN = [0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, -1.0] + [0.0] * 36
_STATE_STD  = ([0.5, 0.5, 0.1,  0.3, 0.3, 0.5,  0.02, 0.02, 0.04]
               + [0.2] * 12 + [2.0] * 12 + [8.0] * 12)
_ACT_MEAN   = [0.0] * 12
_ACT_STD    = [1.0] * 12

_DIM_LABELS = (
    ["vx", "vy", "vz", "wx", "wy", "wz", "gx", "gy", "gz"]
    + [f"q{i}"  for i in range(12)]
    + [f"dq{i}" for i in range(12)]
    + [f"τ{i}"  for i in range(12)]
)


# ── standalone model ─────────────────────────────────────────────────────────

class _StateHead(nn.Module):
    """Single ensemble head: GRU-output(256) → mean(D) + clamped logstd(D)."""

    def __init__(self, hidden: int, state_dim: int, mlp_hidden: int):
        super().__init__()
        self.mean_net   = nn.Sequential(nn.Linear(hidden, mlp_hidden), nn.ELU(),
                                        nn.Linear(mlp_hidden, state_dim))
        self.logstd_net = nn.Sequential(nn.Linear(hidden, mlp_hidden), nn.ELU(),
                                        nn.Linear(mlp_hidden, state_dim))
        # learnable logstd clamp: logstd ∈ [min, min + softplus(delta)]
        self.register_parameter("state_min_logstd",
                                nn.Parameter(torch.zeros(1, state_dim)))
        self.register_parameter("state_log_delta_logstd",
                                nn.Parameter(torch.zeros(1, state_dim)))

    def forward(self, h: torch.Tensor):
        """h: (B, hidden) → mean (B,D), std (B,D)"""
        mean       = self.mean_net(h)
        raw_logstd = self.logstd_net(h)
        lo  = self.state_min_logstd
        hi  = lo + nn.functional.softplus(self.state_log_delta_logstd)
        logstd = lo + (hi - lo) * torch.sigmoid(raw_logstd)
        return mean, torch.exp(logstd)


class EnsembleWorldModel(nn.Module):
    """
    Lightweight re-implementation of SystemDynamicsEnsemble.

    Shared GRU backbone + E independent state prediction heads.
    Hidden state is maintained between forward() calls; call reset() to clear it.
    """

    def __init__(self, state_dim: int, action_dim: int, E: int,
                 rnn_hidden: int, rnn_layers: int, mlp_hidden: int,
                 device: torch.device):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.E          = E
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.device     = device

        input_dim = state_dim + action_dim   # 57
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=rnn_hidden,
                          num_layers=rnn_layers, batch_first=True)
        self.heads = nn.ModuleList(
            [_StateHead(rnn_hidden, state_dim, mlp_hidden) for _ in range(E)]
        )
        self._hidden: torch.Tensor | None = None   # (layers, B, H)

    # ── weight loading from checkpoint ───────────────────────────────────────
    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: torch.device) -> "EnsembleWorldModel":
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd   = ckpt.get("system_dynamics_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        # infer E from state_heads.N keys
        head_ids = {int(k.split(".")[1]) for k in sd if k.startswith("state_heads.")}
        E = max(head_ids) + 1 if head_ids else 1

        # infer rnn_hidden from weight_ih_l0 shape: (3*H, input)
        rnn_hidden = sd["state_base.memory.rnn.weight_ih_l0"].shape[0] // 3
        rnn_layers = sum(1 for k in sd if k.startswith("state_base.memory.rnn.weight_ih_l"))
        state_dim  = sd["state_heads.0.state_mean_layers.2.weight"].shape[0]
        mlp_hidden = sd["state_heads.0.state_mean_layers.0.weight"].shape[0]

        print(f"[model] E={E}  state_dim={state_dim}  rnn_hidden={rnn_hidden}"
              f"  rnn_layers={rnn_layers}  mlp_hidden={mlp_hidden}")

        model = cls(state_dim, 12, E, rnn_hidden, rnn_layers, mlp_hidden, device)

        # map checkpoint keys to our module
        remap: dict[str, str] = {}
        for k in sd:
            if k.startswith("state_base.memory.rnn."):
                remap[k] = k.replace("state_base.memory.rnn.", "rnn.")
            elif k.startswith("state_heads."):
                # state_heads.e.state_mean_layers.L.X  → heads.e.mean_net.L.X
                # state_heads.e.state_logstd_layers.L.X → heads.e.logstd_net.L.X
                # state_heads.e.state_min_logstd        → heads.e.state_min_logstd
                # state_heads.e.state_log_delta_logstd  → heads.e.state_log_delta_logstd
                parts = k.split(".")
                e = parts[1]
                rest = ".".join(parts[2:])
                rest = rest.replace("state_mean_layers.", "mean_net.")
                rest = rest.replace("state_logstd_layers.", "logstd_net.")
                remap[k] = f"heads.{e}.{rest}"

        new_sd = {remap[k]: v for k, v in sd.items() if k in remap}
        missing = set(model.state_dict().keys()) - set(new_sd.keys())
        if missing:
            print(f"[model] WARNING: {len(missing)} keys not loaded: {list(missing)[:4]}")
        model.load_state_dict(new_sd, strict=False)
        model.eval().to(device)
        return model

    # ── stateful inference ───────────────────────────────────────────────────
    def reset(self, batch_size: int | None = None):
        """Clear GRU hidden state (call before a new trajectory)."""
        self._hidden = None

    @torch.no_grad()
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> tuple:
        """
        Args:
            states  : (B, T, state_dim)  normalized
            actions : (B, T, action_dim) normalized
        Returns:
            means   : (E, B, state_dim)
            stds    : (E, B, state_dim)
        After the call the GRU hidden state is updated for the next step.
        """
        x = torch.cat([states, actions], dim=-1)   # (B, T, 57)
        out, self._hidden = self.rnn(x, self._hidden)
        h_last = out[:, -1, :]                      # (B, hidden)

        means_list, stds_list = [], []
        for head in self.heads:
            mu, sigma = head(h_last)
            means_list.append(mu)
            stds_list.append(sigma)

        means = torch.stack(means_list, dim=0)   # (E, B, D)
        stds  = torch.stack(stds_list,  dim=0)   # (E, B, D)
        return means, stds


# ── rollout ───────────────────────────────────────────────────────────────────

def run_rollout(
    model: EnsembleWorldModel,
    ns: torch.Tensor, na: torch.Tensor,
    H: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Single stateful rollout — B=1 (sequential data, no batching needed).

    Step 0  : feed full H-step window to warm up the GRU
    Steps 1+: feed 1 step at a time (hidden state carries context)

    Returns:
        head_mu  : (E, T', D)   per-head predicted next-state (normalized)
        head_sig : (E, T', D)   per-head aleatoric std
        epi_unc  : (T',)        epistemic unc  = std across heads, summed over dims
        ale_unc  : (T',)        aleatoric unc  = mean of per-head stds, summed over dims
    where T' = T - H.
    """
    T  = ns.shape[0]
    T_ = T - H
    E  = model.E

    mu_buf  = np.zeros((T_, E, model.state_dim), dtype=np.float32)
    sig_buf = np.zeros((T_, E, model.state_dim), dtype=np.float32)

    model.reset()

    # warmup: full window, B=1
    s_w = ns[:H].unsqueeze(0)   # (1, H, 45)
    a_w = na[:H].unsqueeze(0)   # (1, H, 12)
    mu, sig = model(s_w, a_w)   # (E, 1, D)
    mu_buf[0]  = mu[:,  0, :].cpu().numpy()
    sig_buf[0] = sig[:, 0, :].cpu().numpy()

    for i in range(1, T_):
        t = H + i
        s_step = ns[t].view(1, 1, -1)
        a_step = na[t].view(1, 1, -1)
        mu, sig = model(s_step, a_step)   # (E, 1, D)
        mu_buf[i]  = mu[:,  0, :].cpu().numpy()
        sig_buf[i] = sig[:, 0, :].cpu().numpy()

        if i % 500 == 0:
            print(f"  step {i}/{T_}")

    # (E, T', D) -> transpose to (E, T', D) — already correct
    head_mu  = mu_buf.transpose(1, 0, 2)   # (E, T', D)  wait, mu_buf is (T',E,D)
    head_sig = sig_buf.transpose(1, 0, 2)  # (E, T', D)

    # epistemic: std across E heads, summed over D
    epi_unc = head_mu.std(axis=0).sum(axis=-1)          # (T',)
    # aleatoric: mean of per-head stds, summed over D
    ale_unc = head_sig.mean(axis=0).sum(axis=-1)        # (T',)

    return head_mu, head_sig, epi_unc, ale_unc


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(E: int):
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("SystemDynamicsEnsemble — Go2 Uncertainty", fontsize=13)
    gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.50, wspace=0.40)

    ax_hist_epi = fig.add_subplot(gs[0, 0:2])
    ax_hist_ale = fig.add_subplot(gs[0, 2:4])
    ax_ts       = fig.add_subplot(gs[0, 4:6])
    ax_mu       = [fig.add_subplot(gs[1, d]) for d in range(6)]
    ax_sig      = [fig.add_subplot(gs[2, d]) for d in range(6)]

    for ax in [ax_hist_epi, ax_hist_ale, ax_ts, *ax_mu, *ax_sig]:
        ax.tick_params(labelsize=6)

    plt.ion()
    plt.show()
    return fig, ax_hist_epi, ax_hist_ale, ax_ts, ax_mu, ax_sig


def update_figure(fig, ax_he, ax_ha, ax_ts,
                  ax_mu, ax_sig,
                  t: int,
                  head_mu: np.ndarray, head_sig: np.ndarray,
                  epi_unc: np.ndarray, ale_unc: np.ndarray,
                  palette, E: int):
    ts = np.arange(t + 1)

    # ── row 0 ────────────────────────────────────────────────────────────
    ax_he.cla()
    ax_he.hist(epi_unc[:t+1], bins=40, color="steelblue", alpha=0.85, edgecolor="none")
    ax_he.set_xlabel("epistemic unc.", fontsize=7)
    ax_he.set_title("Epistemic unc. (history)", fontsize=9)

    ax_ha.cla()
    ax_ha.hist(ale_unc[:t+1], bins=40, color="tomato", alpha=0.85, edgecolor="none")
    ax_ha.set_xlabel("aleatoric unc.", fontsize=7)
    ax_ha.set_title("Aleatoric unc. (history)", fontsize=9)

    ax_ts.cla()
    ax_ts.plot(ts, epi_unc[:t+1], color="steelblue", lw=1.0, label="epistemic")
    ax_ts.plot(ts, ale_unc[:t+1], color="tomato",    lw=1.0, label="aleatoric")
    ax_ts.set_xlabel("step", fontsize=7)
    ax_ts.legend(fontsize=7)
    ax_ts.set_title("Uncertainty over rollout", fontsize=9)

    # ── rows 1-2: per-head per-dim ────────────────────────────────────────
    for d in range(6):
        ax_mu[d].cla()
        ax_sig[d].cla()
        ax_mu[d].set_title(f"μ  {_DIM_LABELS[d]}", fontsize=8)
        ax_sig[d].set_title(f"σ  {_DIM_LABELS[d]}", fontsize=8)
        ax_mu[d].set_xlabel("step", fontsize=6)
        ax_sig[d].set_xlabel("step", fontsize=6)
        for e in range(E):
            c = palette[e]
            ax_mu[d].plot( ts, head_mu[e,  :t+1, d], color=c, lw=0.8, alpha=0.9,
                           label=f"h{e}" if d == 0 else None)
            ax_sig[d].plot(ts, head_sig[e, :t+1, d], color=c, lw=0.8, alpha=0.9)
        if d == 0:
            ax_mu[d].legend(fontsize=6, ncol=3, loc="best")

    fig.canvas.draw()
    fig.canvas.flush_events()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint",   required=True, help="path to .pt checkpoint")
    ap.add_argument("--data",         required=True, help="path to walk_spot2.npz")
    ap.add_argument("--device",       default="cpu")
    ap.add_argument("--pause",        type=float, default=0.05)
    ap.add_argument("--update_every", type=int,   default=10)
    ap.add_argument("--history",      type=int,   default=32,
                    help="GRU warm-up window (default: 32)")
    args = ap.parse_args()

    device = torch.device(args.device)
    H      = args.history

    # ── data ─────────────────────────────────────────────────────────────
    print(f"[data] {args.data}")
    npz = np.load(args.data)
    raw_obs     = torch.tensor(npz["obs"],     dtype=torch.float32, device=device)
    raw_actions = torch.tensor(npz["actions"], dtype=torch.float32, device=device)
    T = raw_obs.shape[0]
    print(f"[data] {T} frames  obs={tuple(raw_obs.shape)}  actions={tuple(raw_actions.shape)}")

    # ── model ─────────────────────────────────────────────────────────────
    model = EnsembleWorldModel.from_checkpoint(args.checkpoint, device)
    E = model.E

    # ── normalize ─────────────────────────────────────────────────────────
    s_mean = torch.tensor(_STATE_MEAN, dtype=torch.float32, device=device)
    s_std  = torch.tensor(_STATE_STD,  dtype=torch.float32, device=device)
    a_mean = torch.tensor(_ACT_MEAN,   dtype=torch.float32, device=device)
    a_std  = torch.tensor(_ACT_STD,    dtype=torch.float32, device=device)
    ns = (raw_obs     - s_mean) / s_std
    na = (raw_actions - a_mean) / a_std

    T_ = T - H
    if T_ <= 0:
        sys.exit(f"Data has {T} frames but --history={H}; need T > H.")

    # ── rollout ───────────────────────────────────────────────────────────
    print(f"[rollout] {T_} steps with E={E} heads ...")
    head_mu, head_sig, epi_unc, ale_unc = run_rollout(model, ns, na, H)
    # head_mu/head_sig: (E, T', 45)
    print(f"[unc] epistemic  min={epi_unc.min():.4f}  max={epi_unc.max():.4f}"
          f"  mean={epi_unc.mean():.4f}")
    print(f"[unc] aleatoric  min={ale_unc.min():.4f}  max={ale_unc.max():.4f}"
          f"  mean={ale_unc.mean():.4f}")

    # ── live viz ──────────────────────────────────────────────────────────
    palette = plt.cm.tab10(np.linspace(0.0, 0.5, E))
    fig, ax_he, ax_ha, ax_ts, ax_mu, ax_sig = make_figure(E)

    print(f"[viz] animating {T_} frames (redraw every {args.update_every} steps)...")
    for t in range(T_):
        if t % args.update_every == 0 or t == T_ - 1:
            update_figure(fig, ax_he, ax_ha, ax_ts, ax_mu, ax_sig,
                          t, head_mu, head_sig, epi_unc, ale_unc, palette, E)
            plt.pause(args.pause)

    plt.ioff()
    print("[viz] done — close the window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
