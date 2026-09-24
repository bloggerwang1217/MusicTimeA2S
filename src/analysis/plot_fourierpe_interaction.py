#!/usr/bin/env python3
"""Gain of each Fourier PE path over the base model on the pre-segmented test sets.

The two paths (Audio Fourier PE on the cross-attention keys/values, Score
Fourier PE on the decoder input) form a 2x2 factorial.  Every arm is drawn as
its paired difference to the base model (no phase model, no path), with
the work-cluster bootstrap 95% CI the tables use, so each path's own gain and
the joint gain read off one axis.  Both test sets share the axes, dodged and
coloured, each against its own base arm, so nothing is compared across test
sets.  The base arm is the zero line; an on/off matrix identifies each PE configuration.

Window set and scoring use ground-truth score MIDI five-bar windows at stride five, restricted to
the windows every arm above evaluated, seed 42 checkpoints throughout.  The bootstrap is delegated to
`paired_work_bootstrap` and its JSON is kept next to the figure statistics;
pass --rescore to recompute it (needed once a new arm's scores land).

    poetry run python src/analysis/plot_fourierpe_interaction.py
    poetry run python src/analysis/plot_fourierpe_interaction.py --rescore --datasets asap --metrics Meter MV2H5
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.paired_work_bootstrap import REPORTED  # noqa: E402

FIGURES = REPO_ROOT / "figures"
STATS = REPO_ROOT / "data/experiments/fourierpe_interaction_stats"


def dataset_inputs(root: Path, title: str) -> dict:
    return {
        "title": title,
        "csv": lambda tag: root / f"eval_{tag}/eval_asap.csv",
        "select": ["--grounding", str(root / "inputs/grounding.jsonl")],
    }


DATASETS = {
    "syn": dataset_inputs(REPO_ROOT / "data/experiments/syn/table1_preseg", "Syn"),
    "asap": dataset_inputs(REPO_ROOT / "data/experiments/asap102/fig2_preseg", "ASAP"),
}

REFERENCE_ARM = "Without phase model"

# arm -> tag of its seed-42 MV2H eval directory.
ARM_TAGS = {
    "Full model": "full_off",
    "Without Fourier PE": "without_fourierpe_off",
    "Without phase model": "without_coordinate_off",
    "Without Duration PE": "without_durationpe_off",
    "Without Audio PE": "without_audiope_off",
}

# Left to right: arm and the label printed under its point.
ORDER = [
    ("Without Fourier PE", "+ Phase\nmodel"),
    ("Without Duration PE", "Audio\nonly"),
    ("Without Audio PE", "Score\nonly"),
    ("Full model", "Both\n(full)"),
]
PATH_ROWS = {
    "Phase supervision": {arm: True for arm, _ in ORDER},
    "Audio Fourier PE": {"Without Fourier PE": False, "Without Duration PE": True,
                         "Without Audio PE": False, "Full model": True},
    "Score Fourier PE": {"Without Fourier PE": False, "Without Duration PE": False,
                         "Without Audio PE": True, "Full model": True},
}
DATASET_COLOR = {"syn": "#1f5fa8", "asap": "#c9662a"}
DATASET_MARKER = {"syn": "o", "asap": "s"}
# Axis label per metric, typeset as the paper writes it.
METRIC_LABEL = {
    "Meter": r"$\Delta F_{\mathrm{meter}}$",
    "MV2H4": r"$\Delta\,\mathrm{MV2H}_4$",
    "MV2H5": r"$\Delta\,\mathrm{MV2H}_5$",
}


def run_bootstrap(dataset: str, out_dir: Path) -> Path:
    """One bootstrap run: the base arm as candidate against every scored arm.
    Only these arms are declared, so the window intersection is the figure's
    own and no other system can drop a window out of it."""
    spec = DATASETS[dataset]
    systems = {arm: spec["csv"](tag) for arm, tag in ARM_TAGS.items()}
    present = {arm: path for arm, path in systems.items() if path.is_file()}
    for arm in systems.keys() - present.keys():
        print(f"missing ({dataset}): {arm} ({systems[arm]})")
    cmd = ["poetry", "run", "python", "-m", "src.analysis.paired_work_bootstrap", "--windows", "intersection",
           *spec["select"], "--candidate", REFERENCE_ARM, "--out", str(out_dir)]
    for arm, path in present.items():
        cmd += ["--system", f"{arm}={path}"]
        if arm != REFERENCE_ARM:
            cmd += ["--reference", arm]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return out_dir / "paired_work_bootstrap.json"


def draw_axes(ax, stats: dict[str, dict], metric: str, marker_scale: float = 1.0) -> list[dict]:
    """All test sets on one axes: per model one point per test set, dodged and
    coloured by test set, each a paired difference to that test set's own base
    arm.  Nothing joins points of different test sets."""
    plotted = []
    datasets = list(stats)
    width = 0.30
    offsets = {d: (i - (len(datasets) - 1) / 2) * width for i, d in enumerate(datasets)}
    xs = list(range(len(ORDER)))
    # The base arm is the zero line, not a point.
    ax.axhline(0.0, color="#626970", linewidth=0.8, zorder=2)
    for dataset in datasets:
        color = DATASET_COLOR[dataset]
        comparisons = stats[dataset]["comparisons"]
        legend = DATASETS[dataset]["title"]
        first = True
        for x, (arm, label) in zip(xs, ORDER):
            xd = x + offsets[dataset]
            pair = comparisons.get(arm, {}).get(metric)
            if pair is None:
                print(f"{dataset} {metric}: no scores for {arm}, point left out")
                continue
            # The bootstrap is candidate (base) minus reference; the figure reads
            # arm minus base, so flip the sign.
            diff = -pair["diff"]
            low, high = -pair["ci95"][1], -pair["ci95"][0]
            ax.errorbar(
                xd, diff, yerr=[[diff - low], [high - diff]],
                fmt=DATASET_MARKER[dataset], markersize=(4.7 if dataset == "syn" else 4.4) * marker_scale,
                markeredgewidth=0.9, markerfacecolor=color,
                color=color, capsize=2.4, capthick=1.2, elinewidth=1.25,
                zorder=3, label=legend if first else None,
            )
            first = False
            plotted.append({"dataset": dataset, "metric": metric, "label": label.replace("\n", " "),
                            "arm": arm, "diff": diff, "ci_low": low, "ci_high": high})
    pending = [x for x, (arm, _) in zip(xs, ORDER)
               if all(arm not in stats[d]["comparisons"] for d in datasets)]
    for x in pending:
        ax.annotate("Pending", xy=(x, 0.5), xycoords=("data", "axes fraction"),
                    ha="center", va="center", fontsize=8, color="#626970")

    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.set_xticks(xs, [""] * len(xs))
    ax.tick_params(axis="x", length=0)
    # Matrix labels use the space below the y axis without narrowing the data area.
    for row, (name, flags) in enumerate(PATH_ROWS.items()):
        y = -0.18 - 0.15 * row
        ax.annotate(name, xy=(-0.23, y), xycoords=("data", "axes fraction"),
                    ha="right", va="center", fontsize=8.2, annotation_clip=False)
        for x, (arm, _) in zip(xs, ORDER):
            ax.annotate("\u2713" if flags[arm] else "\u2717", xy=(x, y),
                        xycoords=("data", "axes fraction"), ha="center", va="center",
                        fontsize=8.5, annotation_clip=False, color="#30363b",
                        fontfamily="DejaVu Sans")
    ax.set_ylabel(METRIC_LABEL.get(metric, f"\u0394 {metric}"), labelpad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#626970")
    ax.spines[["left", "bottom"]].set_linewidth(0.75)
    ax.tick_params(axis="y", length=3, width=0.75, pad=4)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(axis="y", color="#dce0e4", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.margins(y=0.08)
    ax.errorbar([float("nan")], [float("nan")], yerr=[1], fmt="none", color="#626970",
                capsize=2.4, capthick=1.2, elinewidth=1.25,
                label="95% bootstrap CI")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.03, 1.02), ncol=3,
              fontsize=8.5, frameon=False, handletextpad=0.4,
              borderaxespad=0, columnspacing=1.0)
    return plotted


def main() -> None:
    parser = argparse.ArgumentParser(description="Gain of each Fourier PE path over the base model.")
    parser.add_argument("--datasets", nargs="+", default=["syn", "asap"], choices=DATASETS)
    parser.add_argument("--metrics", nargs="+", default=["Meter"], choices=REPORTED)
    parser.add_argument("--rescore", action="store_true", help="rerun the bootstrap even if its JSON exists")
    parser.add_argument("--marker-scale", type=float, default=1.0,
                        help="multiplier on marker size; <1 exposes narrow error bars")
    parser.add_argument("--height", type=float, default=1.7,
                        help="figure height in inches; margins keep their absolute size")
    parser.add_argument("--out", type=Path, default=FIGURES / "fourierpe_interaction",
                        help="output stem; writes <stem>.pdf, <stem>.png and <stem>.csv")
    args = parser.parse_args()

    stats = {}
    for dataset in args.datasets:
        stats_json = STATS / dataset / "paired_work_bootstrap.json"
        if args.rescore or not stats_json.is_file():
            stats_json = run_bootstrap(dataset, STATS / dataset)
        stats[dataset] = json.loads(stats_json.read_text())
        proto = stats[dataset]["protocol"]
        print(f"{dataset}: {proto['windows']} windows, {proto['works']} works, {proto['recordings']} recordings")

    # Times-compatible TrueType text avoids CFF embedding under the Type 42 backend.
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "Liberation Serif", "font.size": 8.5,
        "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "text.color": "#252a30", "axes.labelcolor": "#252a30",
        "xtick.color": "#252a30", "ytick.color": "#252a30",
        "mathtext.fontset": "stix",
    })
    n = len(args.metrics)
    fig, axes = plt.subplots(1, n, figsize=(86 / 25.4 * n, args.height), squeeze=False)
    plotted = []
    for ax, metric in zip(axes[0], args.metrics):
        plotted += draw_axes(ax, stats, metric, marker_scale=args.marker_scale)
    # Absolute margins of the 1.7 in layout (0.5525 in below, 0.2125 in above the axes).
    fig.subplots_adjust(left=0.20 / n, right=0.985, bottom=0.5525 / args.height,
                        top=1 - 0.2125 / args.height, wspace=0.55)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".png"), dpi=400)
    with args.out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "metric", "label", "arm", "diff", "ci_low", "ci_high"])
        writer.writeheader()
        writer.writerows(plotted)
    print(f"wrote {args.out}.pdf/.png/.csv")


if __name__ == "__main__":
    main()
