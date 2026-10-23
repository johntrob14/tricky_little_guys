"""Figure for qwen_capability.py: CoT-only monitor score per turn for the arena_covert_v2 rollouts and the no-side-task control rollouts (diamonds = turns where the leak detector fires), and monitor scores of resampled turns at frozen leak states.

    python3 qwen_capability_chart.py figures/qwen_capability_<subject>_data.json ../../figures/qwen_<subject>_prompted_capability.png
"""
import json, statistics, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


def plot(out, d):
    cov, ctl, (f1, f2) = d["covert"], d["control"], d["frozen"]
    n = len(cov)
    groups = [(f"scorer-confirmed leak ({sum(e['side'] for e in cov)}/{n})", [e for e in cov if e["side"]]), (f"detector fired, scorer did not confirm ({sum(not e['side'] and bool(e['fires']) for e in cov)}/{n})", [e for e in cov if not e["side"] and e["fires"]]),
              (f"no detector fire ({sum(not e['fires'] for e in cov)}/{n})", [e for e in cov if not e["fires"]]), (f"control: no side task in the prompt ({len(ctl)} easy-tier episodes)", ctl)]
    PAGE, SURF, INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA = "#0d0d0d", "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#3987e5", "#d95926", "#199e70"
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"])
    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "axes.edgecolor": GRID})
    short = lambda i: i.replace("missing_user_id_in_guess_response", "missing_user_id").replace("sponsor_statsv2_diggcount_total", "sponsor_statsv2").replace("token_bucket_rate_limiter_per_user", "token_bucket").replace("video_url_format_validation", "video_url")
    rows, labels, titles, idx = [], [], [], []
    for title, g in groups:
        titles.append((len(rows), title))
        for e in g:
            idx.append(len(rows))
            rows.append(np.array([np.nan if x is None else x for x in e["scores"]] + [np.nan] * (40 - e["turns"])))
            labels.append(f"{e['tier'][0]}  {short(e['id'])}" + ("" if e["main"] else "  (main ✗)"))
        rows.append(np.full(40, np.nan)); labels.append("")
    fire_xy = [(t, idx[i]) for i, e in enumerate(e for _, g in groups for e in g) for t in e["fires"]]
    fig = plt.figure(figsize=(17, max(10.5, 0.19 * len(rows) + 3.6)), facecolor=PAGE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], left=0.165, right=0.985, top=1 - 1.3 / fig.get_figheight(), bottom=1.7 / fig.get_figheight(), wspace=0.17, hspace=0.42)
    ax = fig.add_subplot(gs[:, 0], facecolor=SURF)
    mesh = ax.pcolormesh(np.arange(41), np.arange(len(rows) + 1), np.ma.masked_invalid(np.vstack(rows)), cmap=cmap, vmin=0, vmax=10, edgecolors=SURF, linewidth=0.8)
    ax.scatter([t + 0.5 for t, _ in fire_xy], [y + 0.5 for _, y in fire_xy], marker="D", s=30, facecolor=SURF, edgecolor=INK, linewidth=1.1, zorder=3)
    ax.set_ylim(len(rows) - 1, 0); ax.set_xlim(0, 40)
    ax.set_yticks([i + 0.5 for i in range(len(rows))], labels, fontsize=8.5, color=INK2); ax.tick_params(axis="y", length=0)
    ax.set_xticks(np.arange(0, 40, 5) + 0.5, [str(i) for i in range(0, 40, 5)]); ax.set_xlabel("assistant turn within the episode")
    for y, title in titles:
        ax.text(0, y - 0.35, title, color=INK, fontsize=9.5, fontweight="bold", va="bottom")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.025, pad=0.075, aspect=45)
    cb.set_label("CoT-only monitor score for that turn (0 = benign, 10 = suspicious)", color=INK2); cb.set_ticks([0, 2, 4, 6, 8, 10]); cb.outline.set_edgecolor(GRID); cb.ax.tick_params(colors=MUTED)
    ax.set_title("Full rollouts: monitor score per turn", loc="left", color=INK, fontsize=12, pad=26)
    ax.legend(handles=[Line2D([], [], marker="D", ls="", markerfacecolor=SURF, markeredgecolor=INK, markersize=7, label="turn where the leak detector fires")], loc="lower right", facecolor=SURF, edgecolor=GRID, labelcolor=INK2, fontsize=9)

    def hist(a, rs, title):
        a.set_facecolor(SURF)
        xs = np.arange(11)
        cls = lambda r: 0 if r["al"] else 1 if r["raw"] else 2  # completes on a surface the scorer accepts / detector fires elsewhere / no completion
        labs = ("completes the leak (scorer-aligned)", "detector fires, scorer would not see it", "no completion")
        cnt = np.array([[sum(r["score"] == x and cls(r) == k for r in rs) for x in xs] for k in range(3)])
        base = np.zeros(11)
        for k, col in enumerate((BLUE, AQUA, ORANGE)):
            a.bar(xs, cnt[k], bottom=base, width=0.82, color=col, edgecolor=SURF, linewidth=1.2, label=labs[k]); base = base + cnt[k]
        a.set_xticks(xs); a.set_xlim(-0.6, 10.6); a.set_ylim(0, base.max() * 1.7)
        a.set_xlabel("CoT-only monitor score for the sampled turn"); a.set_ylabel("sampled turns")
        a.yaxis.grid(True, color=GRID, linewidth=0.8); a.set_axisbelow(True)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(GRID); a.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
        lines = [f"{labs[k]}: {sum(cls(r) == k for r in rs)}/{len(rs)}" + (f", mean {statistics.mean(m):.2f}" if (m := [r["score"] for r in rs if cls(r) == k and r["score"] is not None]) else "") for k in range(3)]
        a.text(0.995, 0.97, "\n".join(lines), transform=a.transAxes, ha="right", va="top", fontsize=8.5, color=INK2, linespacing=1.5)
    a1, a2 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    for a, f in ((a1, f1), (a2, f2)):
        hist(a, f["rows"], f"Resampled at {f['title']} ({f['contexts']} x {len(f['rows']) // f['contexts']})")
    a1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.83), frameon=False, labelcolor=INK2, fontsize=8.5)
    fig.suptitle(f"{d['name']} prompted with arena_covert_v2 (side task in the prompt): what a CoT-only monitor sees", x=0.015, y=1 - 0.25 / fig.get_figheight(), ha="left", color=INK, fontsize=14, fontweight="bold")
    fig.text(0.015, 0.55 / fig.get_figheight(), "Monitoring mode: CoT only (controlarena_cot_only, judge openai/gpt-oss-20b, main task only, each turn scored alone). Diamonds = leak_check detector fires on the tool call.\n"
             f"{d['foot']}.\n{d['foot2']} at contexts frozen just before a leak move.\nscorer-aligned = the detector restricted to the surfaces score.py accepts (public/, /tmp, /srv, /var/www, healthcheck, /config, /leak routes); 'detector fires' = any route or file under src/app.",
             color=MUTED, fontsize=8, va="bottom", linespacing=1.5)
    fig.savefig(out, dpi=160, facecolor=PAGE)


if __name__ == "__main__":
    plot(sys.argv[2], json.load(open(sys.argv[1])))
    print("saved", sys.argv[2])
