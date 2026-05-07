"""Plot the 3-way bench (raw_mp4 vs wds_default vs wds_optimized) and the
S3 download bench. Output dark editorial SVGs that drop into the website.

Usage:
    plot_3way.py --in-dir results/ --out-dir plots/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

INK = "#0d0e0c"; INK_DEEP = "#07080a"
PAPER = "#f1ead7"; RULE = "#312f29"
TEXT = "#ede4ce"; TEXT_SOFT = "#b9b09a"; TEXT_MUTE = "#7a7363"
CINNABAR = "#e8513b"; CINNABAR_DK = "#b13c2c"
LIME = "#cce15c"; GOLD = "#d8a64a"; TEAL = "#4a8a86"; INDIGO = "#5c6bc0"

FORMAT_COLOR = {
    "raw_mp4":       CINNABAR,
    "wds_default":   GOLD,
    "wds_optimized": LIME,
    "clips":         CINNABAR,
    "shards":        LIME,
}
FORMAT_LABEL = {
    "raw_mp4":       "raw mp4 (per-clip + parquet)",
    "wds_default":   "WDS, default encode (GOP=250, bf=2)",
    "wds_optimized": "WDS, optimized encode (GOP=32, bf=0, CFR)",
    "clips":         "tiny mp4 clips (1 file each)",
    "shards":        "1-GB WebDataset shards",
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
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       TEXT_MUTE,
    "ytick.color":       TEXT_MUTE,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "grid.color":        RULE,
    "grid.alpha":        0.4,
    "legend.facecolor":  INK_DEEP,
    "legend.edgecolor":  RULE,
    "legend.labelcolor": TEXT,
    "legend.fontsize":   10,
    "legend.frameon":    False,
    "font.family":       ["JetBrains Mono", "DejaVu Sans Mono", "monospace"],
    "font.size":         10,
    "lines.linewidth":   2.4,
}


def _style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, color=PAPER, loc="left", pad=12,
                     fontfamily="JetBrains Mono", fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel.upper(), labelpad=10, fontsize=9, color=TEXT_MUTE)
    if ylabel:
        ax.set_ylabel(ylabel.upper(), labelpad=10, fontsize=9, color=TEXT_MUTE)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(RULE)


def plot_3way(json_path: Path, out_path: Path):
    data = json.loads(json_path.read_text())
    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    workers_seen = sorted({int(w) for fmt in data["results"].values() for w in fmt})

    for fmt in ["raw_mp4", "wds_default", "wds_optimized"]:
        rs = data["results"].get(fmt, {})
        xs, ys = [], []
        for w in workers_seen:
            r = rs.get(str(w))
            if r and "samples_per_s" in r:
                xs.append(w)
                ys.append(r["samples_per_s"])
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", color=FORMAT_COLOR[fmt],
                markerfacecolor=FORMAT_COLOR[fmt], markeredgecolor=INK,
                markersize=9, label=FORMAT_LABEL[fmt])
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 8),
                        textcoords="offset points", ha="center",
                        color=PAPER, fontsize=9, fontfamily="JetBrains Mono")

    # GPU ceiling line at 64 sps
    ax.axhline(64, color=INDIGO, linestyle=":", linewidth=1.4, alpha=0.65)
    ax.text(workers_seen[-1] - 0.2, 65,
            "L40S trainer ceiling (64 sps)",
            color=INDIGO, fontsize=9, fontfamily="JetBrains Mono",
            ha="right", va="bottom")

    ax.set_xticks(workers_seen)
    ax.set_xlim(min(workers_seen) - 0.5, max(workers_seen) + 0.5)
    ax.set_ylim(bottom=0, top=max(70, max(
        (r["samples_per_s"] for fmt in data["results"].values()
         for r in fmt.values() if "samples_per_s" in r),
        default=20,
    )) * 1.12)
    _style(ax,
           title=f"3-way dataloader throughput on g6e.8xlarge (32 vCPU, L40S, T={data['T']}, batch={data['batch_size']})",
           xlabel=f"num_workers · 32 vCPU available",
           ylabel="samples / second")
    ax.grid(axis="y", linestyle=":", color=RULE, alpha=0.5)
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_s3_download(json_path: Path, out_path: Path):
    if not json_path.exists():
        print(f"skip {json_path} (does not exist)")
        return
    data = json.loads(json_path.read_text())
    fig, ax = plt.subplots(figsize=(10.5, 5.0))

    for fmt in ["clips", "shards"]:
        if fmt not in data["tests"]:
            continue
        rows = data["tests"][fmt]
        xs = [r["conc"] for r in rows]
        ys = [r["mb_per_s"] for r in rows]
        ax.plot(xs, ys, marker="o", color=FORMAT_COLOR[fmt],
                markerfacecolor=FORMAT_COLOR[fmt], markeredgecolor=INK,
                markersize=9, label=FORMAT_LABEL[fmt])
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f} MB/s", xy=(x, y), xytext=(0, 8),
                        textcoords="offset points", ha="center",
                        color=PAPER, fontsize=9, fontfamily="JetBrains Mono")

    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 16, 32, 64])
    ax.set_xticklabels(["4", "16", "32", "64"])
    ax.set_ylim(bottom=0)
    _style(ax,
           title=f"S3 cold-cache download throughput — small clips vs 1-GB shards",
           xlabel="aws cli concurrent requests",
           ylabel="effective download throughput (MB/s)")
    ax.grid(axis="y", linestyle=":", color=RULE, alpha=0.5)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {out_path}")


def write_summary(thr_path: Path, dl_path: Path, out_path: Path):
    summary = {}
    if thr_path.exists():
        d = json.loads(thr_path.read_text())
        for fmt, rs in d["results"].items():
            best = None
            for w_str, r in rs.items():
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
    if dl_path.exists():
        d = json.loads(dl_path.read_text())
        summary["download"] = d.get("tests", {})
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLT_RC)
    if (args.in_dir / "throughput_3way.json").exists():
        plot_3way(args.in_dir / "throughput_3way.json",
                  args.out_dir / "throughput_3way.svg")
    if (args.in_dir / "s3_download.json").exists():
        plot_s3_download(args.in_dir / "s3_download.json",
                         args.out_dir / "s3_download.svg")
    write_summary(args.in_dir / "throughput_3way.json",
                  args.in_dir / "s3_download.json",
                  args.out_dir / "summary_3way.json")


if __name__ == "__main__":
    main()
