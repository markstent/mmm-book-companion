"""Chapter 34 lab: write Sedge's one-page readout from the case study's saved results."""

import json
import sys
from pathlib import Path

from mmm_book.experiment import EXPERIMENT_ROAS

RESULTS = Path(__file__).resolve().parent.parent / "results" / "part9.json"
NAMES = {"tv": "Television", "search": "Search", "meta": "Meta", "youtube": "YouTube", "ooh": "OOH"}


def readout(results: dict) -> str:
    """The model card and readout as Markdown, from the numbers in ``part9.json``."""
    final = results["validation"]["final"]
    marginal = results["marginal"]
    plan = results["optimisation"]["plans"]["operational"]
    gain = plan["gain_m"]
    meta = results["validation"]["holdout"]
    lines = [
        "# Sedge Shoes: what we got for the 6 million",
        "",
        f"Media earned {final['total']['roas']:.2f} pounds of revenue per pound spent "
        f"({final['total']['contribution_m']:.1f}m on {final['total']['spend_m']:.1f}m).",
        "",
        "## Recommendation",
        "",
        "Hold the budget at 6.0m and move money as below (annual, m).",
        "",
        "| Channel | Now | Plan | Average ROAS (90%) | Next pound |",
        "|---|---|---|---|---|",
    ]
    for ch, name in NAMES.items():
        row = final[ch]
        lines.append(
            f"| {name} | {row['spend_m']:.2f} | {plan['plan_m'][ch]:.2f} "
            f"| {row['mean']:.2f} ({row['lo']:.2f} to {row['hi']:.2f}) "
            f"| {marginal[ch]['marginal']['mean']:.2f} |"
        )
    lines += [
        "",
        f"Expected gain {gain['mean'] * 1000:,.0f}k revenue a year "
        f"(90% {gain['lo'] * 1000:,.0f}k to {gain['hi'] * 1000:,.0f}k), "
        f"{plan['profit_m'] * 1000:,.0f}k profit.",
        "The probability that the plan beats the current mix is a statement about the model only.",
        "",
        "## What to be careful about",
        "",
        f"- Meta's holdout measured its average return, {EXPERIMENT_ROAS:.2f} "
        f"(the model's estimate for the test window is {meta['test_roas_after']['mean']:.2f}), "
        "not the next pound, so the Meta cut is a judgement about the slope.",
        f"- Only {marginal['youtube']['p_marginal_above_break_even']:.0%} of draws put YouTube's "
        "next pound above the 2.0 break-even.",
        "",
        "## Measure next",
        "",
        "Ranked by what the answer would change: "
        + ", ".join(NAMES[ch] for ch in results["recommendation"]["voi_ranking"])
        + ".",
        "",
    ]
    return "\n".join(lines)


def main(path: Path = RESULTS) -> str:
    text = readout(json.loads(Path(path).read_text()))
    print(text)
    return text


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:]))
