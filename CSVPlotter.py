#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

DATA_DIR = "res/CNHexpcsv"
DISTANCES = list(range(100, 201, 10))  # 100,110,...,200 (11 files)
BINS = 32                               # cnh__hist_bin_0..31
GRID = (4, 4)                           # 4x4 zones -> a0..a15

def zone_index(row: int, col: int) -> int:
    return row * GRID[1] + col  # a = row*4 + col

def load_csv(distance: int) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{distance}mm.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    # Fast parse; only keep columns we need (hist bins for all a0..a15)
    usecols = []
    for b in range(BINS):
        for a in range(GRID[0] * GRID[1]):
            usecols.append(f"cnh__hist_bin_{b}_a{a}")
    return pd.read_csv(path, usecols=usecols, low_memory=False)

def extract_hist(df: pd.DataFrame, a: int) -> np.ndarray:
    cols = [f"cnh__hist_bin_{b}_a{a}" for b in range(BINS)]
    # Average over time/rows; ignore NaNs
    return np.nanmean(df[cols].to_numpy(dtype=float), axis=0)

def main():
    # Preload all CSVs (fail fast with a helpful message if any missing)
    data = {}
    for d in DISTANCES:
        try:
            data[d] = load_csv(d)
        except Exception as e:
            print(e)
            return

    # Matplotlib figure: 3x4 grid for 11 plots (last cell hidden)
    fig, axes = plt.subplots(3, 4, figsize=(12, 7), constrained_layout=True)
    axes = axes.ravel()

    # Space at bottom for sliders
    plt.subplots_adjust(bottom=0.18)

    # Initial selection
    row0, col0 = 0, 0
    a0 = zone_index(row0, col0)
    x = np.arange(BINS)

    lines = []
    for i, d in enumerate(DISTANCES):
        ax = axes[i]
        y = extract_hist(data[d], a0)
        (ln,) = ax.plot(x, y, lw=1.5)
        lines.append(ln)
        ax.set_title(f"{d} mm", fontsize=10)
        ax.set_xlim(0, BINS - 1)
        # Optional: same y-scale across panels for easier comparison
        ax.set_ylim(bottom=0)  # autoscale top later after first draw
        ax.grid(True, alpha=0.3)
    # Hide the unused 12th subplot
    axes[-1].axis("off")

    # Uniform y-limits (max across current selection)
    ymax = max(ln.get_ydata().max() for ln in lines)
    for ax in axes[:-1]:
        ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1)

    fig.suptitle("CNH 4×4 Cell Histogram vs Distance (select ROW/COL)", fontsize=12)

    # Slider axes
    ax_row = plt.axes([0.10, 0.08, 0.35, 0.03])
    ax_col = plt.axes([0.55, 0.08, 0.35, 0.03])

    s_row = Slider(ax=ax_row, label="ROW (0–3)", valmin=0, valmax=3, valinit=row0, valstep=1)
    s_col = Slider(ax=ax_col, label="COL (0–3)", valmin=0, valmax=3, valinit=col0, valstep=1)

    def update(_):
        a = zone_index(int(s_row.val), int(s_col.val))
        ymax_local = 0.0
        for ln, d in zip(lines, DISTANCES):
            y = extract_hist(data[d], a)
            ln.set_ydata(y)
            ymax_local = max(ymax_local, np.nanmax(y) if np.isfinite(np.nanmax(y)) else 0.0)
        # Rescale y-axis uniformly for fair comparison
        lim = ymax_local * 1.05 if ymax_local > 0 else 1.0
        for ax in axes[:-1]:
            ax.set_ylim(0, lim)
        fig.canvas.draw_idle()

    s_row.on_changed(update)
    s_col.on_changed(update)

    # Keyboard nudge (optional quality-of-life)
    def on_key(event):
        nonlocal_row, nonlocal_col = int(s_row.val), int(s_col.val)
        if event.key in ("up", "down", "left", "right"):
            if event.key == "up":    nonlocal_row = max(0, nonlocal_row - 1)
            if event.key == "down":  nonlocal_row = min(3, nonlocal_row + 1)
            if event.key == "left":  nonlocal_col = max(0, nonlocal_col - 1)
            if event.key == "right": nonlocal_col = min(3, nonlocal_col + 1)
            s_row.set_val(nonlocal_row)
            s_col.set_val(nonlocal_col)

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

if __name__ == "__main__":
    main()
