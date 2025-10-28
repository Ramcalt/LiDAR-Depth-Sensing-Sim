import csv
from pathlib import Path
from typing import Union, Iterable, Tuple, List, Dict, Optional
import matplotlib.pyplot as plt

def _read_nonempty_lines(source: Union[str, Path, Iterable[str]]) -> List[str]:
    """Return all non-empty lines from a file path, a multi-line string, or an iterable of lines."""
    if isinstance(source, (str, Path)) and Path(str(source)).exists():
        with open(source, "r", newline="") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    elif isinstance(source, (str, Path)):
        # Treat as a (possibly multi-line) string
        lines = [ln.strip() for ln in str(source).splitlines() if ln.strip()]
    else:
        # Iterable of lines
        lines = [ln.strip() for ln in source if ln.strip()]
    if not lines:
        raise ValueError("No non-empty CSV rows found in 'source'.")
    return lines

def _parse_zone_from_row(row_str: str, zone_index: int, bins_per_zone: int) -> Dict[str, object]:
    """Parse a single CSV row and extract status, distance, ambient, and CNH bins for one zone."""
    reader = csv.reader([row_str], skipinitialspace=True)
    tokens = next(reader, [])
    if len(tokens) < 2:
        raise ValueError("CSV row must start with SensorMode, MessageType.")

    fields_per_zone = 3 + bins_per_zone
    header_offset = 2
    total_needed = header_offset + fields_per_zone * 64
    if len(tokens) < total_needed:
        raise ValueError(
            f"Row too short for 64 zones with {bins_per_zone} bins. "
            f"Expected ≥ {total_needed}, got {len(tokens)}."
        )
    if not (0 <= zone_index < 64):
        raise IndexError("zone_index must be in [0, 63].")

    start = header_offset + zone_index * fields_per_zone
    status = int(tokens[start + 0])
    distance_mm = int(tokens[start + 1])
    ambient = float(tokens[start + 2])
    bins = [float(v) for v in tokens[start + 3 : start + 3 + bins_per_zone]]

    return {
        "status": status,
        "distance_mm": distance_mm,
        "ambient": ambient,
        "bins": bins,
    }

def plot_cnh_zone_all_rows(
    source: Union[str, Path, Iterable[str]],
    zone_index: int,
    *,
    bins_per_zone: int = 18,
    ax: Optional[plt.Axes] = None,
    show_legend: bool = True,
    limit_rows: Optional[int] = None,
) -> Tuple[plt.Axes, List[Dict[str, object]]]:
    """
    Plot CNH bins for 'zone_index' for every CSV row present in 'source' on the same axes.

    Parameters
    ----------
    source : str | Path | Iterable[str]
        - Path to a CSV file containing multiple one-row frames (one per line), or
        - A multi-line string, or
        - Any iterable of lines.
    zone_index : int
        Zone to plot in [0, 63].
    bins_per_zone : int, optional
        Number of CNH bins per zone. Default 18 (Bins[0..17]).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure and axes are created.
    show_legend : bool, optional
        Whether to show a legend identifying each row. Default True.
    limit_rows : int | None, optional
        If provided, only the first 'limit_rows' non-empty lines are plotted.

    Returns
    -------
    (ax, rows_info) : (matplotlib.axes.Axes, list[dict])
        The axes used and a list of per-row metadata dicts:
        {
            "row_index": int,
            "status": int,
            "distance_mm": int,
            "ambient": float,
            "bins": list[float]
        }
    """
    lines = _read_nonempty_lines(source)
    if limit_rows is not None:
        lines = lines[:max(0, int(limit_rows))]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    rows_info: List[Dict[str, object]] = []
    x = list(range(bins_per_zone))

    for idx, row in enumerate(lines):
        try:
            info = _parse_zone_from_row(row, zone_index, bins_per_zone)
        except Exception as exc:
            # Skip malformed rows but continue plotting others
            print(f"[warn] Skipping row {idx}: {exc}")
            continue

        rows_info.append({"row_index": idx, **info})
        label = f"row {idx} | st={info['status']} d={info['distance_mm']}mm amb={info['ambient']:.2f}"
        ax.plot(x, info["bins"], marker="o", linewidth=1.5, label=label)

    ax.set_xlabel("CNH bin")
    ax.set_ylabel("Value")
    ax.set_title(f"Zone {zone_index} — CNH over {len(rows_info)} row(s)")
    ax.set_xticks(x)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    if show_legend and rows_info:
        ax.legend(loc="best", fontsize="small", ncol=1, frameon=True)

    return ax, rows_info

# ---- Example usage ----
ax, rows_info = plot_cnh_zone_all_rows("frame.csv", zone_index=5)
plt.show()
#
# Or with a string that contains several rows (each row is one frame):
# multi_row = "SM_TOF2_SCANNING, MT_TOF2_SCAN, 0, 1678, ...\nSM_TOF2_SCANNING, MT_TOF2_SCAN, 0, 1677, ..."
# ax, rows_info = plot_cnh_zone_all_rows(multi_row, zone_index=0)
# plt.show()
