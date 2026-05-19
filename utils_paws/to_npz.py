#!/usr/bin/env python3
"""
Convert a go2_data_recorder .bin file to a .npz file.

Frame layout (236 bytes, packed, matches C++ struct Frame):
  double  timestamp       [1]
  float   obs[45]         [45]
    obs[0:3]   base linear velocity   (m/s)
    obs[3:6]   base angular velocity  (rad/s)
    obs[6:9]   projected gravity
    obs[9:21]  joint positions        (rad)
    obs[21:33] joint velocities       (rad/s)
    obs[33:45] joint torques          (Nm)
  float   action[12]      [12]  commanded joint positions (rad)

Usage:
  python3 to_npz.py walk1.bin
  python3 to_npz.py walk1.bin -o walk1.npz   # explicit output path
"""

import argparse
import struct
import sys
import numpy as np
from pathlib import Path

JOINT_NAMES = [
    "FR_Hip", "FR_Thigh", "FR_Calf",
    "FL_Hip", "FL_Thigh", "FL_Calf",
    "RR_Hip", "RR_Thigh", "RR_Calf",
    "RL_Hip", "RL_Thigh", "RL_Calf",
]

# struct Frame { double ts; float obs[45]; float action[12]; }
FRAME_FMT  = "<d45f12f"
FRAME_SIZE = struct.calcsize(FRAME_FMT)
assert FRAME_SIZE == 236, f"Frame size mismatch: {FRAME_SIZE}"


def load_bin(path: str):
    with open(path, "rb") as f:
        n_bytes = f.read(8)
        if len(n_bytes) < 8:
            sys.exit("File too short — not a valid recording.")
        (n,) = struct.unpack("<Q", n_bytes)
        print(f"  frames declared : {n}")

        raw = f.read(n * FRAME_SIZE)

    actual = len(raw) // FRAME_SIZE
    if actual < n:
        print(f"  WARNING: only {actual} complete frames found (expected {n}), truncating.")
        n = actual

    timestamps = np.empty(n, dtype=np.float64)
    obs        = np.empty((n, 45), dtype=np.float32)
    actions    = np.empty((n, 12), dtype=np.float32)

    for i in range(n):
        chunk = raw[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]
        vals  = struct.unpack_from(FRAME_FMT, chunk)
        timestamps[i] = vals[0]
        obs[i]        = vals[1:46]
        actions[i]    = vals[46:58]

    return timestamps, obs, actions


def main():
    ap = argparse.ArgumentParser(description="Convert .bin recording to .npz")
    ap.add_argument("bin_file", help="Input .bin file")
    ap.add_argument("-o", "--output", default=None, help="Output .npz path")
    args = ap.parse_args()

    out_path = args.output or str(Path(args.bin_file).with_suffix(".npz"))

    print(f"Loading {args.bin_file} ...")
    timestamps, obs, actions = load_bin(args.bin_file)
    n = len(timestamps)

    duration = timestamps[-1] - timestamps[0] if n > 1 else 0.0
    dt_mean  = duration / (n - 1) if n > 1 else 0.0
    hz_mean  = 1.0 / dt_mean if dt_mean > 0 else 0.0

    print(f"  frames loaded   : {n}")
    print(f"  duration        : {duration:.2f} s")
    print(f"  mean rate       : {hz_mean:.1f} Hz")

    np.savez(
        out_path,
        timestamps        = timestamps,          # (N,)
        obs               = obs,                 # (N, 45)
        actions           = actions,             # (N, 12)
        # convenient named slices
        base_lin_vel      = obs[:, 0:3],         # (N, 3)
        base_ang_vel      = obs[:, 3:6],         # (N, 3)
        projected_gravity = obs[:, 6:9],         # (N, 3)
        joint_pos         = obs[:, 9:21],        # (N, 12)
        joint_vel         = obs[:, 21:33],       # (N, 12)
        joint_torque      = obs[:, 33:45],       # (N, 12)
    )
    print(f"\nSaved -> {out_path}")
    print("\nArrays inside the .npz:")
    print(f"  timestamps        {timestamps.shape}   float64  seconds since epoch")
    print(f"  obs               {obs.shape}  float32  full 45-dim observation")
    print(f"  actions           {actions.shape}  float32  commanded joint positions")
    print(f"  base_lin_vel      (N, 3)   float32  m/s")
    print(f"  base_ang_vel      (N, 3)   float32  rad/s")
    print(f"  projected_gravity (N, 3)   float32")
    print(f"  joint_pos         (N,12)   float32  rad   order: {', '.join(JOINT_NAMES)}")
    print(f"  joint_vel         (N,12)   float32  rad/s")
    print(f"  joint_torque      (N,12)   float32  Nm")
    print("\nTo load in Python:")
    print(f"  data = np.load('{out_path}')")
    print( "  joint_pos = data['joint_pos']   # shape (N, 12)")


if __name__ == "__main__":
    main()
