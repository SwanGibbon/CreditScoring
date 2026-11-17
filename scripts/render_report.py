"""Render lightweight, deterministic portfolio SVGs from the public summary."""
import argparse
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(summary_path, directory, preview=None):
    report = json.loads(Path(summary_path).read_text())
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    intervals = {r["policy"]: r for r in report["auc_intervals"]}
    selected = intervals[report["primary_policy"]]
    delta = report["paired_auc_differences"][0]["delta_auc"]
    source_rows = report["dataset"].get("known_row_counts", {})
    history_rows = sum(v for k, v in source_rows.items() if not k.startswith("application_"))
    data_label = f"{len(report['dataset']['inputs'])} CSV files"
    if history_rows: data_label += f" · {history_rows / 1e6:.0f}M historical rows"
    tests = report.get("verification", {}).get("tests_passed")
    test_label = f"{tests} tests passed" if tests else "Reproducible run"
    split_label = f"{report['dataset']['holdout_rows']:,} holdout applications · {report['config']['outer_folds']} outer folds"
    hero = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="320" viewBox="0 0 1100 320" role="img" aria-labelledby="title desc">
<title id="title">Home Credit Default Risk</title>
<desc id="desc">Verified independent holdout AUC {selected['auc']:.4f}; history contribution +{delta:.4f}. Three LightGBM feature variants with nested evaluation.</desc>
<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#101d32"/><stop offset="1" stop-color="#1c3445"/></linearGradient></defs>
<rect width="1100" height="320" rx="18" fill="url(#bg)"/>
<rect x="36" y="32" width="5" height="66" rx="2" fill="#5eead4"/>
<g font-family="Inter, Segoe UI, Arial, sans-serif">
<text x="57" y="64" fill="#f8fafc" font-size="35" font-weight="700">Home Credit Default Risk</text>
<text x="57" y="94" fill="#cbd5e1" font-size="17">Relational features · nested evaluation · independent holdout</text>
<text x="57" y="124" fill="#94a3b8" font-size="13">{escape(data_label)} · LightGBM same-model ablation</text>
<g fill="#142639" stroke="#365064"><rect x="36" y="157" width="330" height="119" rx="12"/><rect x="385" y="157" width="330" height="119" rx="12"/><rect x="734" y="157" width="330" height="119" rx="12"/></g>
<g fill="#94a3b8" font-size="11" font-weight="600" letter-spacing="1.2"><text x="57" y="181">INDEPENDENT HOLDOUT AUC</text><text x="406" y="181">ADDED HISTORY SIGNAL</text><text x="755" y="181">VERIFICATION</text></g>
<g fill="#5eead4" font-size="36" font-weight="700"><text x="57" y="226">{selected['auc']:.4f}</text><text x="406" y="226">+{delta:.4f}</text></g>
<text x="755" y="226" fill="#f8fafc" font-size="32" font-weight="700">{escape(test_label)}</text>
<g fill="#cbd5e1" font-size="12"><text x="57" y="253">95% CI [{selected['lower_95']:.4f}, {selected['upper_95']:.4f}]</text><text x="406" y="253">over the engineered application baseline</text><text x="755" y="253">{escape(split_label)}</text></g>
<text x="36" y="303" fill="#94a3b8" font-size="11">Measured benchmark · Seed 42 · Run {escape(report.get('run_date_utc') or 'date not recorded')} · Conditional bootstrap intervals</text>
</g></svg>'''
    (directory / "overview.svg").write_text(hero + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.hashsalt": "creditscoring-report-v1", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(10.5, 3.6), layout="constrained")
    keys = ["raw_lightgbm", "application_lightgbm", "history_lightgbm"]
    labels = ["Raw application", "Engineered application", "Application + history"]
    colors = ["#64748b", "#0284c7", "#0f766e"]
    for y, key, color in zip([2, 1, 0], keys, colors):
        r = intervals[key]
        ax.errorbar(r["auc"], y, xerr=[[r["auc"]-r["lower_95"]], [r["upper_95"]-r["auc"]]],
                    fmt="o", color=color, capsize=5, markersize=8, linewidth=2)
        ax.text(r["upper_95"]+.0014, y, f"{r['auc']:.4f}", va="center", color=color, weight="bold")
    ax.set(yticks=[2, 1, 0], yticklabels=labels, xlim=(.735, .817), ylim=(-.55, 2.55),
           xlabel="ROC-AUC on the independent holdout")
    ax.set_title("One model family. Three feature sets.", loc="left", fontsize=15, weight="bold", pad=22)
    ax.text(0, 1.01, f"{report['dataset']['holdout_rows']:,} applications · 95% stratified bootstrap intervals · {report['bootstrap']['replicates']:,} replicates",
            transform=ax.transAxes, fontsize=9, color="#64748b")
    ax.grid(axis="x", color="#e2e8f0", linewidth=.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=9)
    fig.savefig(directory / "holdout-auc.svg", metadata={"Date": None, "Creator": "CreditScoring"})
    svg_path = directory / "holdout-auc.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    if preview: fig.savefig(preview, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", default="reports/home-credit-full-001.json")
    parser.add_argument("--directory", default="docs/assets")
    parser.add_argument("--preview")
    args = parser.parse_args()
    render(args.summary, args.directory, args.preview)
