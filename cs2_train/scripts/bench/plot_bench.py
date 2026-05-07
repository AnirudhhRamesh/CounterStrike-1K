"""Render benchmark JSONs to dark-editorial SVGs that drop straight into the
On-Feeding-the-GPU article (dataloader-2026/plots/).

Usage:
    uv run python -m cs2_train.scripts.bench.plot_bench \
        --in-dir  /opt/dlami/nvme/bench/results \
        --out-dir /opt/dlami/nvme/bench/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Editorial palette (matches /Users/anirudhh/projects/learning/dataloader-2026/style.css).
INK         = "#0d0e0c"
INK_DEEP    = "#07080a"
PAPER       = "#f1ead7"
PAPER_DIM   = "#b9b09a"
RULE        = "#312f29"
TEXT        = "#ede4ce"
TEXT_SOFT   = "#b9b09a"
TEXT_MUTE   = "#7a7363"
CINNABAR    = "#e8513b"
CINNABAR_DK = "#b13c2c"
LIME        = "#cce15c"
GOLD        = "#d8a64a"
TEAL        = "#4a8a86"
INDIGO      = "#5c6bc0"

STAGE_COLORS = {
    # raw mp4 stages
    "open":         CINNABAR,
    "seek+decode":  GOLD,
    "materialise":  TEAL,
    "actions":      LIME,
    # wds extras
    "shard_open":   CINNABAR_DK,
    "tar_extract":  INDIGO,
}

FORMAT_COLORS = {
    "raw_mp4":    CINNABAR,
    "webdataset": LIME,
}

PLT_RC = {
    "figure.facecolor":  INK,
    "axes.facecolor":    INK,
    "savefig.facecolor": INK,
    "savefig.edgecolor": INK,
    "axes.edgecolor":    RULE,
    "axes.labelcolor":   TEXT_SOFT,
    "axes.titlecolor":   PAPER,
    "axes.titlesize":    13,
    "axes.titleweight":  "regular",
    "axes.titlepad":     16,
    "axes.labelsize":    10,
    "axes.labelweight":  "regular",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       TEXT_MUTE,
    "ytick.color":       TEXT_MUTE,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "grid.color":        RULE,
    "grid.linestyle":    "-",
    "grid.alpha":        0.4,
    "legend.facecolor":  INK_DEEP,
    "legend.edgecolor":  RULE,
    "legend.labelcolor": TEXT,
    "legend.fontsize":   10,
    "legend.frameon":    False,
    "font.family":       ["JetBrains Mono", "DejaVu Sans Mono", "monospace"],
    "font.size":         10,
    "lines.linewidth":   2.0,
    "patch.linewidth":   0,
}

def _apply_rc():
    plt.rcParams.update(PLT_RC)

def _style_axes(ax, *, title=None, xlabel=None, ylabel=None, mono_title=False):
    if title:
        ax.set_title(title, color=PAPER, loc="left", pad=12,
                     fontfamily="JetBrains Mono", fontsize=11,
                     fontweight="regular")
    if xlabel:
        ax.set_xlabel(xlabel.upper(), labelpad=10, fontsize=9, color=TEXT_MUTE)
    if ylabel:
        ax.set_ylabel(ylabel.upper(), labelpad=10, fontsize=9, color=TEXT_MUTE)
    ax.tick_params(axis="both", which="both", length=4, width=1)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(RULE)
        ax.spines[spine].set_linewidth(1)


def plot_io_breakdown(io_path: Path, out: Path):
    data = json.loads(io_path.read_text())
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Bars: raw_mp4 cold, raw_mp4 warm, webdataset.
    bars = []  # list of (label, dict-of-stage->ms)

    def _med(d, key):
        s = d.get(key, {})
        return s.get("median") if s.get("n", 0) > 0 else None

    raw = data.get("raw_mp4", {})
    if raw:
        cold_open = _med(raw, "open_cold_ms")
        cold_sd   = _med(raw, "seek_decode_cold_ms")
        warm_open = _med(raw, "open_warm_ms")
        warm_sd   = _med(raw, "seek_decode_warm_ms")
        mat       = _med(raw, "materialize_ms") or 0.0
        act       = _med(raw, "actions_ms") or 0.0
        if cold_open is not None and cold_sd is not None:
            bars.append(("raw mp4 — cold (first-touch)", {
                "open": cold_open, "seek+decode": cold_sd,
                "materialise": mat, "actions": act,
            }))
        if warm_sd is not None:
            bars.append(("raw mp4 — warm (decoder cached)", {
                "open": warm_open or 0.0, "seek+decode": warm_sd,
                "materialise": mat, "actions": 0.0,
            }))

    wds = data.get("webdataset", {})
    if wds and "decode_ms" in wds:
        wds_d = {
            "shard_open":   wds["shard_open_ms"]["median"],
            "tar_extract":  wds["tar_extract_ms"]["median"],
            "seek+decode":  wds["decode_ms"]["median"],
            "materialise":  wds["materialize_ms"]["median"],
            "actions":      wds["actions_ms"]["median"],
        }
        bars.append(("webdataset (in-mem mp4 + parquet bytes)", wds_d))

    labels = [b[0] for b in bars]
    y = list(range(len(bars)))[::-1]  # top bar = first row

    stage_order = ["open", "shard_open", "tar_extract", "seek+decode", "materialise", "actions"]

    for yi, (_, stages) in zip(y, bars):
        x_left = 0.0
        for stage in stage_order:
            v = stages.get(stage, 0.0)
            if v <= 0:
                continue
            ax.barh(yi, v, left=x_left, height=0.62,
                    color=STAGE_COLORS[stage],
                    edgecolor=INK, linewidth=0.0,
                    label=stage if stage not in {h.get_label() for h in ax.containers if h.get_label() in STAGE_COLORS} else None)
            # Annotate inside bar if wide enough
            if v > 1.5:
                ax.text(x_left + v / 2, yi, f"{v:.1f}", ha="center", va="center",
                        color=INK, fontsize=8, fontweight="bold",
                        fontfamily="JetBrains Mono")
            x_left += v
        ax.text(x_left + 0.6, yi, f"= {x_left:.1f} ms", ha="left", va="center",
                color=PAPER, fontsize=9, fontfamily="JetBrains Mono")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=TEXT, fontsize=10, fontfamily="JetBrains Mono")
    ax.invert_yaxis()
    _style_axes(ax,
                xlabel="median per-sample wall time (ms)",
                title=f"FIG · 5 — Per-stage timing, T={data.get('T', 8)}, n={data.get('n_samples', 200)}")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=False, nbins=8))
    ax.grid(axis="x", linestyle=":", color=RULE, alpha=0.4)

    # Custom legend
    handles = []
    seen = set()
    for stage in stage_order:
        if any(stage in b[1] and b[1][stage] > 0 for b in bars) and stage not in seen:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=STAGE_COLORS[stage], label=stage))
            seen.add(stage)
    ax.legend(handles=handles, loc="lower right", ncol=3, fontsize=9, frameon=False,
              labelcolor=TEXT, handlelength=1.3, handleheight=1.0, columnspacing=1.6)

    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {out}")


def plot_throughput(thr_path: Path, out: Path):
    data = json.loads(thr_path.read_text())
    fig, ax = plt.subplots(figsize=(10, 5.0))

    workers_seen = sorted({int(k) for fmt in data["results"].values() for k in fmt.keys()})

    for fmt, results in data["results"].items():
        xs, ys, ys_low = [], [], []
        for w in workers_seen:
            r = results.get(str(w))
            if not r or "samples_per_s" not in r:
                continue
            xs.append(w)
            ys.append(r["samples_per_s"])
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=2.4, markersize=8,
                color=FORMAT_COLORS[fmt], markerfacecolor=FORMAT_COLORS[fmt],
                markeredgecolor=INK, label=fmt)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}", xy=(x, y), xytext=(0, 8),
                        textcoords="offset points",
                        ha="center", color=PAPER, fontsize=9,
                        fontfamily="JetBrains Mono")

    ax.set_xticks(workers_seen)
    ax.set_xlim(min(workers_seen) - 0.4, max(workers_seen) + 0.4)
    ax.set_ylim(bottom=0)
    _style_axes(ax,
                xlabel="num_workers (= 8 vCPU on this host)",
                ylabel="samples / second",
                title=f"FIG · 6 — End-to-end throughput, T={data.get('T', 8)}, batch={data.get('batch_size', 16)}")
    ax.grid(axis="y", linestyle=":", color=RULE, alpha=0.5)
    ax.axvline(8, color=CINNABAR_DK, linestyle=":", linewidth=1, alpha=0.4)
    ax.text(8.12, ax.get_ylim()[1] * 0.96, "vCPU = 8", color=TEXT_MUTE,
            fontsize=8, fontfamily="JetBrains Mono", va="top")
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {out}")


def plot_T_sweep(t_path: Path, out: Path):
    data = json.loads(t_path.read_text())
    fig, (ax_sps, ax_fps) = plt.subplots(1, 2, figsize=(11, 4.6))

    Ts_seen = sorted({int(k) for fmt in data["results"].values() for k in fmt.keys()})

    for fmt, results in data["results"].items():
        xs, sps, fps = [], [], []
        for T in Ts_seen:
            r = results.get(str(T))
            if not r or "samples_per_s" not in r:
                continue
            xs.append(T)
            sps.append(r["samples_per_s"])
            fps.append(r.get("frames_per_s", r["samples_per_s"] * T))

        ax_sps.plot(xs, sps, marker="o", linewidth=2.2, markersize=7,
                    color=FORMAT_COLORS[fmt], markerfacecolor=FORMAT_COLORS[fmt],
                    markeredgecolor=INK, label=fmt)
        ax_fps.plot(xs, fps, marker="s", linewidth=2.2, markersize=7,
                    color=FORMAT_COLORS[fmt], markerfacecolor=FORMAT_COLORS[fmt],
                    markeredgecolor=INK, label=fmt)
        for x, y in zip(xs, sps):
            ax_sps.annotate(f"{y:.0f}", xy=(x, y), xytext=(0, 8),
                            textcoords="offset points", ha="center",
                            color=PAPER, fontsize=8, fontfamily="JetBrains Mono")
        for x, y in zip(xs, fps):
            ax_fps.annotate(f"{y:.0f}", xy=(x, y), xytext=(0, 8),
                            textcoords="offset points", ha="center",
                            color=PAPER, fontsize=8, fontfamily="JetBrains Mono")

    for ax in (ax_sps, ax_fps):
        ax.set_xticks(Ts_seen)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle=":", color=RULE, alpha=0.5)
        ax.legend(loc="best", frameon=False)
    _style_axes(ax_sps, xlabel="window length T (frames)", ylabel="samples / s",
                title=f"FIG · 7a — Throughput vs T (samples/s)")
    _style_axes(ax_fps, xlabel="window length T (frames)", ylabel="frames / s",
                title=f"FIG · 7b — Throughput vs T (frames/s)")

    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {out}")


def plot_window_latency(lat_path: Path, out: Path):
    data = json.loads(lat_path.read_text())
    fig, ax = plt.subplots(figsize=(10, 4.6))

    Ts = sorted([int(k) for k in data["results"].keys()])
    medians = [data["results"][str(T)]["median"] for T in Ts]
    p95     = [data["results"][str(T)]["p95"] for T in Ts]
    p05     = [data["results"][str(T)]["p05"] for T in Ts]

    ax.fill_between(Ts, p05, p95, color=GOLD, alpha=0.18, label="p05 – p95")
    ax.plot(Ts, medians, marker="o", linewidth=2.4, markersize=7,
            color=GOLD, markerfacecolor=GOLD, markeredgecolor=INK,
            label="median")
    for T, y in zip(Ts, medians):
        ax.annotate(f"{y:.1f} ms", xy=(T, y), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    color=PAPER, fontsize=9, fontfamily="JetBrains Mono")

    # Linear fit on the slope from T=4 to T=max
    if len(Ts) >= 2:
        import numpy as np
        idx = [i for i, T in enumerate(Ts) if T >= 4]
        if len(idx) >= 2:
            xs = [Ts[i] for i in idx]
            ys = [medians[i] for i in idx]
            m, b = np.polyfit(xs, ys, 1)
            xfit = [Ts[idx[0]], Ts[idx[-1]]]
            yfit = [m * x + b for x in xfit]
            ax.plot(xfit, yfit, linestyle="--", color=CINNABAR, linewidth=1.2, alpha=0.8,
                    label=f"slope ≈ {m:.2f} ms / frame")

    ax.set_xticks(Ts)
    ax.set_ylim(bottom=0)
    _style_axes(ax,
                xlabel="window length T (frames)",
                ylabel="single-process per-window latency (ms)",
                title=f"FIG · 8 — Cold-vs-per-frame: where the time really goes (n={data.get('n_samples', 200)})")
    ax.grid(axis="y", linestyle=":", color=RULE, alpha=0.5)
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {out}")


def write_summary_json(throughput_path: Path, t_sweep_path: Path,
                       io_path: Path, out: Path):
    """Compact recap JSON for the in-page table."""
    summary = {}

    if throughput_path.exists():
        thr = json.loads(throughput_path.read_text())
        for fmt, results in thr["results"].items():
            best = None
            for w_str, r in results.items():
                if "samples_per_s" not in r:
                    continue
                if best is None or r["samples_per_s"] > best["samples_per_s"]:
                    best = {**r, "num_workers": int(w_str)}
            if best:
                summary[fmt] = {
                    "best_samples_per_s": best["samples_per_s"],
                    "best_workers":       best["num_workers"],
                    "best_ms_per_batch":  best["ms_per_batch"],
                }

    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir",  required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_rc()

    if (in_dir / "io_breakdown.json").exists():
        plot_io_breakdown(in_dir / "io_breakdown.json", out_dir / "io_breakdown.svg")
    if (in_dir / "throughput.json").exists():
        plot_throughput(in_dir / "throughput.json", out_dir / "throughput_vs_workers.svg")
    if (in_dir / "T_sweep.json").exists():
        plot_T_sweep(in_dir / "T_sweep.json", out_dir / "throughput_vs_T.svg")
    if (in_dir / "window_latency.json").exists():
        plot_window_latency(in_dir / "window_latency.json", out_dir / "window_latency.svg")

    write_summary_json(
        in_dir / "throughput.json",
        in_dir / "T_sweep.json",
        in_dir / "io_breakdown.json",
        out_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
