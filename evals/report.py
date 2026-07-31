#!/usr/bin/env python3
"""Render one self-contained HTML report from one or more summary.json files.

  python evals/report.py <summary.json> [<summary.json> ...] -o report.html

Per-model resolve rate + Wilson 95% CI (computed over all (task, repeat) pairs),
a failure-class ledger, a per-task matrix, and a two-model delta table when two
summaries are supplied. The HTML contains only logical model ids and task ids:
no absolute paths, no usernames, no endpoints.

Only the Python standard library is used.
"""

import argparse
import html
import json
import math
import sys
from pathlib import Path


def wilson_ci(k, n, z=1.96):
    if n <= 0:
        return [0.0, 1.0]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def load_summary(path):
    data = json.loads(Path(path).read_text())
    return {
        "model": data["model"],
        "suite": data.get("suite", ""),
        "resolved": int(data["resolved"]),
        "total": int(data["total"]),
        "per_task": data.get("per_task", []),
    }


def _all_runs(summary):
    for task in summary.get("per_task", []):
        for r in task.get("runs", []):
            yield r


def failure_ledger(summaries):
    counts = {}
    for s in summaries:
        for r in _all_runs(s):
            fc = r.get("fail_class")
            key = fc if fc else "resolved"
            counts[key] = counts.get(key, 0) + 1
    return counts


def per_task_matrix(summaries):
    rows = {}
    for s in summaries:
        for task in s["per_task"]:
            rows.setdefault(task["id"], {})[s["model"]] = task["resolve_frac"]
    return rows


def two_model_delta(summaries):
    if len(summaries) != 2:
        return None
    a, b = summaries
    matrix = per_task_matrix(summaries)
    deltas = []
    for task_id in sorted(matrix):
        ra = matrix[task_id].get(a["model"])
        rb = matrix[task_id].get(b["model"])
        d = None
        if ra is not None and rb is not None:
            d = round(rb - ra, 4)
        deltas.append((task_id, ra, rb, d))
    overall = round(b["resolved"] / b["total"] - a["resolved"] / a["total"], 4) \
        if a["total"] and b["total"] else 0.0
    return {"a": a["model"], "b": b["model"], "per_task": deltas, "overall": overall}


def _fmt_pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def _fmt_ci(ci):
    return f"{ci[0]:.0%}\u2013{ci[1]:.0%}"


def render_html(summaries):
    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>"]
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>charlie-code eval report</title>")
    parts.append("<style>")
    parts.append(
        "body{font:14px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;"
        "margin:24px;color:#1b2430;background:#f3f5f7}"
        ".wrap{max-width:900px;margin:0 auto}"
        "h1{font-size:20px;margin:0 0 14px}"
        "h2{font-size:15px;margin:22px 0 8px;border-bottom:1px solid #d9dfe6;padding-bottom:5px}"
        "table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0}"
        "th,td{border:1px solid #d9dfe6;padding:6px 9px;text-align:left}"
        "th{background:#eef1f5}"
        "td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}"
        ".ok{color:#187f37}.bad{color:#cf222e}.muted{color:#5b6774}"
    )
    parts.append("</style></head><body><div class='wrap'>")
    parts.append("<h1>charlie-code eval report</h1>")

    parts.append("<h2>Per-model resolve rate (95% Wilson CI)</h2>")
    parts.append("<table><tr><th>model</th><th class='num'>resolved</th>"
                 "<th class='num'>total</th><th class='num'>resolve rate</th>"
                 "<th class='num'>95% CI</th></tr>")
    for s in summaries:
        ci = wilson_ci(s["resolved"], s["total"])
        rate = s["resolved"] / s["total"] if s["total"] else 0.0
        parts.append(
            f"<tr><td>{html.escape(s['model'])}</td>"
            f"<td class='num'>{s['resolved']}</td><td class='num'>{s['total']}</td>"
            f"<td class='num'>{rate:.1%}</td><td class='num'>{_fmt_ci(ci)}</td></tr>"
        )
    parts.append("</table>")

    ledger = failure_ledger(summaries)
    parts.append("<h2>Failure-class ledger</h2>")
    parts.append("<table><tr><th>class</th><th class='num'>count</th></tr>")
    for key in ["resolved", "wrong_answer", "step_limit", "env_error", "infra"]:
        if key in ledger:
            cls = "ok" if key == "resolved" else "bad"
            parts.append(
                f"<tr><td class='{cls}'>{html.escape(key)}</td>"
                f"<td class='num'>{ledger[key]}</td></tr>"
            )
    for key, count in sorted(ledger.items()):
        if key in ("resolved", "wrong_answer", "step_limit", "env_error", "infra"):
            continue
        parts.append(
            f"<tr><td>{html.escape(str(key))}</td><td class='num'>{count}</td></tr>"
        )
    parts.append("</table>")

    matrix = per_task_matrix(summaries)
    parts.append("<h2>Per-task resolve matrix</h2>")
    parts.append("<table><tr><th>task</th>"
                 + "".join(f"<th class='num'>{html.escape(s['model'])}</th>" for s in summaries)
                 + "</tr>")
    for task_id in sorted(matrix):
        cells = "".join(
            f"<td class='num'>{_fmt_pct(matrix[task_id].get(s['model']))}</td>"
            for s in summaries
        )
        parts.append(f"<tr><td>{html.escape(task_id)}</td>{cells}</tr>")
    parts.append("</table>")

    delta = two_model_delta(summaries)
    if delta:
        parts.append("<h2>Two-model delta</h2>")
        parts.append("<table><tr><th>task</th>"
                     f"<th class='num'>{html.escape(delta['a'])}</th>"
                     f"<th class='num'>{html.escape(delta['b'])}</th>"
                     "<th class='num'>delta (B - A)</th></tr>")
        for task_id, ra, rb, d in delta["per_task"]:
            dcell = "n/a" if d is None else f"{d:+.0%}"
            dcls = ""
            if d is not None:
                dcls = "ok" if d > 0 else ("bad" if d < 0 else "muted")
            parts.append(
                f"<tr><td>{html.escape(task_id)}</td>"
                f"<td class='num'>{_fmt_pct(ra)}</td><td class='num'>{_fmt_pct(rb)}</td>"
                f"<td class='num {dcls}'>{dcell}</td></tr>"
            )
        parts.append(
            f"<tr><td><b>overall</b></td>"
            f"<td class='num'>{_fmt_pct(summaries[0]['resolved']/summaries[0]['total'] if summaries[0]['total'] else 0)}</td>"
            f"<td class='num'>{_fmt_pct(summaries[1]['resolved']/summaries[1]['total'] if summaries[1]['total'] else 0)}</td>"
            f"<td class='num'><b>{delta['overall']:+.0%}</b></td></tr>"
        )
        parts.append("</table>")

    parts.append("</div></body></html>")
    return "".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render an eval report HTML")
    parser.add_argument("summaries", nargs="+", help="one or more summary.json paths")
    parser.add_argument("-o", "--out", required=True, help="output HTML path")
    args = parser.parse_args(argv)

    summaries = [load_summary(p) for p in args.summaries]
    html_text = render_html(summaries)
    Path(args.out).write_text(html_text)
    print(f"wrote {args.out} ({len(summaries)} summary file(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
