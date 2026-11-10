"""The numbers Part IX of the book prints (Chapters 34 to 39 and Appendix F).

Every estimate in the Sedge case study is what the case-study model reports
when fitted to the synthetic data. These tests recompute them from the five
case-study fits (``mmm_book.case_results``) and check them against the
values the book prints, rounded the way the book rounds them, and against
``results/part9.json``, which the book's Part IX figures read.

The fits are cached in ``.cache``; on a fresh checkout the first run fits
them, which takes several minutes.
"""

import json

import pytest

from mmm_book.case_results import MARGIN, RESULTS_PATH, part9_results
from mmm_book.sedge import CHANNELS, GROUND_TRUTH


@pytest.fixture(scope="module")
def r():
    return part9_results()


def close(x, target, tol):
    return x == pytest.approx(target, abs=tol)


def rounded(summary, digits=2):
    """(mean, lo, hi) of a posterior summary, rounded the way the book prints it."""
    return tuple(round(summary[k], digits) for k in ("mean", "lo", "hi"))


FITS = ("model1", "model2", "model2_cal", "model2_train130", "model2_train104")


def all_diagnostics(r):
    return {
        "model1": r["ch35"]["model1"]["diagnostics"],
        "model2": r["ch35"]["model2"]["diagnostics"],
        "model2_cal": r["ch36"]["holdout"]["diagnostics"],
        "model2_train130": r["ch36"]["validation"]["diagnostics"],
        "model2_train104": r["ch36"]["validation"]["diagnostics_104"],
    }


def test_results_file_matches_a_fresh_computation(r):
    saved = json.loads(RESULTS_PATH.read_text())

    def walk(a, b, path=""):
        if isinstance(a, dict):
            assert set(a) == set(b), path
            for k in a:
                walk(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, list):
            assert len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                walk(x, y, f"{path}[{i}]")
        elif isinstance(a, float):
            assert a == pytest.approx(b, rel=1e-6, abs=1e-9), path
        else:
            assert a == b, path

    walk(r, saved)


def test_chapter34_data(r):
    c = r["ch34"]
    assert close(c["corr_meta_youtube_national"], 0.85, 0.005)
    assert close(c["corr_meta_youtube_panel"], 0.76, 0.005)
    assert close(c["corr_search_category"], 0.72, 0.005)
    assert close(c["annual_revenue_m"], 47.9, 0.05)


def test_chapter35_first_model_and_repair(r):
    c = r["ch35"]
    assert close(c["prior_ooh_share_above_20m"], 0.026, 0.0005)
    assert close(c["prior_search_share_halflife_above_8"], 0.020, 0.0005)
    m1 = c["model1"]
    assert m1["diagnostics"]["divergences"] == 0
    assert m1["diagnostics"]["max_rhat"] <= 1.01
    assert close(m1["r2"], 0.93, 0.005)
    assert close(m1["wmape"], 0.051, 0.0005)
    s = m1["roas"]["search"]
    assert (round(s["mean"], 1), round(s["lo"], 1), round(s["hi"], 1)) == (4.7, 3.9, 5.5)
    assert close(s["contribution_m"], 6.6, 0.05)
    m2 = c["model2"]
    assert m2["diagnostics"]["divergences"] == 0
    s = m2["roas"]["search"]
    assert (round(s["mean"], 1), round(s["lo"], 1), round(s["hi"], 1)) == (2.2, 1.7, 2.8)
    assert close(c["meta_m"]["mean"], 2.3, 0.05)
    assert close(c["youtube_m"]["mean"], 1.9, 0.05)
    assert close(c["pair_m"]["mean"], 4.1, 0.06)
    assert close(c["pair_corr"], -0.45, 0.005)


def test_chapter36_validation_and_calibration(r):
    c = r["ch36"]
    v = c["validation"]
    assert close(v["test_wmape"], 0.047, 0.0005)
    assert close(v["mean_error_in_launch"], 117_000, 1_000)
    assert close(v["mean_error_outside_launch"], 13_000, 1_000)
    for d in (v["diagnostics"], v["diagnostics_104"], c["holdout"]["diagnostics"]):
        assert d["divergences"] == 0
        assert d["max_rhat"] <= 1.01
    h = c["holdout"]
    assert (round(h["meta_before"]["mean"], 2), round(h["meta_before"]["lo"], 2)) == (1.52, 0.90)
    assert round(h["meta_before"]["hi"], 2) == 2.29
    assert (round(h["meta_after"]["mean"], 2), round(h["meta_after"]["lo"], 2)) == (1.65, 1.05)
    assert round(h["meta_after"]["hi"], 2) == 2.36
    # Appendix H (Figure H.2): "about 18 in 100" above the 2.0x break-even
    assert round(h["meta_after_p_above_break_even"] * 100) == 18
    assert close(h["lift_before"]["mean"], 97_000, 1_000)
    assert close(h["lift_before_sd"], 26_000, 1_000)
    assert close(h["lift_after"]["mean"], 107_000, 1_000)
    assert close(1 - h["meta_width_after"] / h["meta_width_before"], 0.06, 0.01)
    assert close(h["youtube_after"]["mean"], 2.20, 0.005)
    final = c["final"]
    assert close(final["total"]["contribution_m"], 12.43, 0.005)
    printed = {"search": 2.22, "meta": 1.65, "youtube": 2.20, "tv": 2.57, "ooh": 1.65}
    for ch, value in printed.items():
        assert round(final[ch]["mean"], 2) == value, ch
    assert close(c["profit_m"]["total"], 0.21, 0.005)
    d = c["decomposition_m"]
    assert close(d["baseline"]["mean"], 33.7, 0.05)
    assert close(d["controls"]["mean"], 1.8, 0.05)
    assert close(d["promotion"]["mean"], 0.9, 0.05)
    assert close(d["price"]["mean"], 0.7, 0.05)
    assert close(d["media"]["mean"], 12.4, 0.05)
    assert abs(d["residual"]["mean"]) < 0.05
    inside = [ch for ch, t in c["truth"].items() if t["inside_interval"]]
    assert sorted(inside) == ["meta", "ooh", "search", "youtube"]


def test_chapter37_marginal_returns(r):
    c = r["ch37"]
    printed = {"search": 1.39, "meta": 1.11, "youtube": 1.87, "tv": 2.22, "ooh": 1.39}
    for ch, value in printed.items():
        assert round(c[ch]["marginal"]["mean"], 2) == value, ch
    assert c["tv"]["p_marginal_above_break_even"] > 0.99
    assert round(c["youtube"]["p_marginal_above_break_even"] * 100) == 39
    for ch in ("search", "meta", "ooh"):
        assert c[ch]["p_marginal_above_break_even"] < 0.015


def test_chapter38_plans_and_scenarios(r):
    p = r["ch38"]["plans"]
    assert close(p["naive_40"]["gain_m"]["mean"], 0.58, 0.005)
    assert close(p["naive_40"]["p_gain"], 0.97, 0.005)
    assert close(p["bounded_optimiser"]["gain_m"]["mean"], 0.277, 0.002)
    op = p["operational"]
    assert close(op["gain_m"]["mean"], 0.266, 0.002)
    assert close(op["gain_m"]["lo"], 0.11, 0.005)
    assert close(op["gain_m"]["hi"], 0.44, 0.005)
    assert op["p_gain"] > 0.99
    assert close(op["profit_m"], 0.133, 0.002)
    assert close(p["ch27_unconstrained"]["p_gain"], 0.94, 0.005)
    assert close(p["ch27_constrained"]["p_gain"], 0.99, 0.005)
    s = r["ch38"]["scenarios"]
    assert close(s["cut15"]["difference_m"]["mean"], -1.52, 0.005)
    assert close(s["cut15"]["profit_change_m"], 0.14, 0.005)
    assert close(s["up10"]["difference_m"]["mean"], 0.96, 0.005)
    assert close(s["up10"]["profit_change_m"], -0.12, 0.005)
    assert close(s["targeted_cut"]["difference_m"]["mean"], -1.18, 0.005)
    assert close(s["targeted_cut"]["profit_change_m"], 0.31, 0.005)


def test_chapter27_uncertainty_from_the_fitted_model(r):
    c = r["ch27"]
    assert close(c["gain_m"]["mean"], 0.45, 0.005)
    assert close(c["gain_m"]["lo"], 0.12, 0.005)
    assert close(c["gain_m"]["hi"], 0.81, 0.005)
    assert close(c["p_gain"], 0.99, 0.005)
    assert close(c["p_gain_below_100k"], 0.04, 0.01)
    pct = c["resolved_percentiles_m"]
    assert pct["tv"][0] == pytest.approx(1.82, abs=0.01)  # at the ceiling in every draw
    assert pct["meta"][3] == pytest.approx(1.20, abs=0.01)  # at the floor in most draws
    assert pct["youtube"][4] - pct["youtube"][0] > 0.4  # YouTube moves most


def test_chapter39_value_of_information(r):
    c = r["ch39"]
    assert c["voi_ranking"][:2] == ["meta", "youtube"]
    assert c["voi_ranking"][-1] == "tv"
    assert close(c["voi"]["meta"]["moved_m"], 0.54, 0.01)
    assert close(c["voi"]["youtube"]["moved_m"], 0.35, 0.01)
    assert c["voi"]["tv"]["moved_m"] < 0.1


# ---- Appendix F, "Reproducing the results" -----------------------------------

MODEL1_PRINTED = {
    "search": (4.68, 3.91, 5.52),
    "meta": (1.28, 0.76, 1.95),
    "youtube": (1.42, 0.65, 2.33),
    "tv": (2.21, 2.05, 2.38),
    "ooh": (1.59, 1.12, 2.16),
}
MODEL2_PRINTED = {
    "search": (2.23, 1.73, 2.77),
    "meta": (1.52, 0.90, 2.29),
    "youtube": (2.34, 1.36, 3.45),
    "tv": (2.57, 2.41, 2.74),
    "ooh": (1.65, 1.14, 2.28),
}
CALIBRATED_PRINTED = {
    "search": (2.22, 1.71, 2.77),
    "meta": (1.65, 1.05, 2.36),
    "youtube": (2.20, 1.24, 3.27),
    "tv": (2.57, 2.40, 2.74),
    "ooh": (1.65, 1.16, 2.29),
}


def test_appendix_f_results_table(r):
    m1 = r["ch35"]["model1"]["roas"]
    m2 = r["ch35"]["model2"]["roas"]
    cal = r["ch36"]["final"]
    for ch in CHANNELS:
        assert rounded(m1[ch]) == MODEL1_PRINTED[ch], ch
        assert rounded(m2[ch]) == MODEL2_PRINTED[ch], ch
        assert rounded(cal[ch]) == CALIBRATED_PRINTED[ch], ch
    # Television's lower ends sit just above the true 2.40
    assert round(m2["tv"]["lo"], 3) == 2.408
    assert round(cal["tv"]["lo"], 3) == 2.404
    # eight of the ten Model 2 and calibrated intervals contain the truth;
    # the two that miss are Television's
    misses = [
        (fit, ch)
        for fit, block in (("model2", m2), ("cal", cal))
        for ch in CHANNELS
        if not block[ch]["lo"] <= GROUND_TRUTH[ch] <= block[ch]["hi"]
    ]
    assert misses == [("model2", "tv"), ("cal", "tv")]
    # Model 1's Television interval misses the truth on the low side
    assert m1["tv"]["hi"] < GROUND_TRUTH["tv"]
    # "within 3 per cent of the true 12.72 million"
    true_total = sum(t["contribution_m"] for t in r["ch36"]["truth"].values())
    assert round(true_total, 3) == 12.715  # printed as 12.72 (half up)
    assert 0 < 1 - cal["total"]["contribution_m"] / true_total < 0.03


def test_chapter35_model1_table(r):
    m1 = r["ch35"]["model1"]["roas"]
    printed = {
        "search": (4.7, 3.9, 5.5),
        "meta": (1.3, 0.8, 1.9),
        "youtube": (1.4, 0.6, 2.3),
        "tv": (2.2, 2.0, 2.4),
        "ooh": (1.6, 1.1, 2.2),
    }
    for ch, value in printed.items():
        assert rounded(m1[ch], 1) == value, ch


def test_fit_and_sampler_diagnostics(r):
    m1, m2 = r["ch35"]["model1"], r["ch35"]["model2"]
    # Model 2 brings Search back "with almost no change in fit" (R squared 0.93 again)
    assert round(m2["r2"], 2) == 0.93
    assert round(m2["wmape"], 3) == 0.052
    assert abs(m2["wmape"] - m1["wmape"]) < 0.002
    diags = all_diagnostics(r)
    assert set(diags) == set(FITS)
    for name, d in diags.items():
        assert d["divergences"] == 0, name
        assert d["draws"] == 8000, name
        assert d["max_rhat"] <= 1.01, name
        # "bulk and tail effective sample sizes above 800 for every parameter"
        assert min(d["min_ess_bulk"], d["min_ess_tail"]) > 800, name
    # Model 1: "every R-hat at 1.003 or below"
    assert round(m1["diagnostics"]["max_rhat"], 3) == 1.003
    # Appendix F: "the largest is 1.009", and it is Model 2's
    largest = max(diags, key=lambda k: diags[k]["max_rhat"])
    assert largest == "model2"
    assert round(diags["model2"]["max_rhat"], 3) == 1.009


# ---- Chapter 35: contributions -------------------------------------------------


def test_chapter35_meta_youtube_contributions(r):
    c = r["ch35"]
    assert rounded(c["meta_m"], 1) == (2.3, 1.4, 3.4)
    assert rounded(c["youtube_m"], 1) == (1.9, 1.1, 2.8)
    assert rounded(c["pair_m"], 1) == (4.1, 3.3, 5.3)
    assert round(c["pair_corr"], 2) == -0.45
    assert round(c["model1"]["roas"]["search"]["contribution_m"], 1) == 6.6
    assert round(c["model1"]["r2"], 2) == 0.93
    assert round(c["model1"]["wmape"] * 100, 1) == 5.1


# ---- Chapter 36 ----------------------------------------------------------------


def test_chapter36_folds(r):
    folds = r["ch36"]["folds"]
    assert set(folds) == {"104", "130", "156"}

    def span(ch):
        values = [folds[k][ch] for k in folds]
        return round(min(values), 2), round(max(values), 2)

    assert span("tv") == (2.57, 2.63)
    assert span("youtube") == (1.61, 2.34)
    moves = {
        ch: max(folds[k][ch] for k in folds) - min(folds[k][ch] for k in folds) for ch in CHANNELS
    }
    # Search, Meta and OOH "each moving by about 0.2 to 0.3"
    for ch in ("search", "meta", "ooh"):
        assert round(moves[ch], 1) in (0.2, 0.3), ch
    assert max(moves, key=moves.get) == "youtube"
    assert min(moves, key=moves.get) == "tv"


def test_chapter36_validation(r):
    v = r["ch36"]["validation"]
    assert v["train_weeks"] == 130
    assert v["launch_week"] == 141
    assert round(v["test_wmape"] * 100, 1) == 4.7
    assert round(v["mean_error_outside_launch"], -2) == 13_300
    assert round(v["mean_error_in_launch"], -3) == 117_000


def test_chapter36_model2_table(r):
    m2 = r["ch36"]["model2"]
    printed = {"search": 3.12, "meta": 2.28, "youtube": 1.87, "tv": 3.60, "ooh": 1.48}
    for ch, value in printed.items():
        assert round(m2[ch]["contribution_m"], 2) == value, ch
        assert rounded(m2[ch]) == MODEL2_PRINTED[ch], ch
    assert round(m2["total"]["contribution_m"], 2) == 12.35
    assert round(m2["total"]["roas"], 2) == 2.06
    assert m2 == r["ch35"]["model2"]["roas"]


def test_chapter36_final_table(r):
    final = r["ch36"]["final"]
    printed = {"search": 3.11, "meta": 2.48, "youtube": 1.76, "tv": 3.60, "ooh": 1.49}
    for ch, value in printed.items():
        assert round(final[ch]["contribution_m"], 2) == value, ch
        assert (
            round(final[ch]["spend_m"], 2)
            == {"search": 1.40, "meta": 1.50, "youtube": 0.80, "tv": 1.40, "ooh": 0.90}[ch]
        )
    assert round(final["total"]["contribution_m"], 2) == 12.43
    assert round(final["total"]["spend_m"], 2) == 6.00
    assert round(final["total"]["roas"], 2) == 2.07
    # calibration moves about 0.2 million towards Meta and a little away from YouTube
    m2 = r["ch36"]["model2"]
    assert round(final["meta"]["contribution_m"] - m2["meta"]["contribution_m"], 1) == 0.2
    assert final["youtube"]["contribution_m"] < m2["youtube"]["contribution_m"]


def test_chapter36_holdout_in_text(r):
    h = r["ch36"]["holdout"]
    assert rounded(h["meta_before"]) == (1.52, 0.90, 2.29)
    assert rounded(h["meta_after"]) == (1.65, 1.05, 2.36)
    assert rounded(h["youtube_before"]) == (2.34, 1.36, 3.45)
    assert rounded(h["youtube_after"]) == (2.20, 1.24, 3.27)
    # Appendix F: 97,000 (59,000 to 143,000), then 107,000 (70,000 to 147,000)
    assert tuple(round(x, -3) for x in rounded(h["lift_before"], 0)) == (97_000, 59_000, 143_000)
    assert tuple(round(x, -3) for x in rounded(h["lift_after"], 0)) == (107_000, 70_000, 147_000)
    assert round(h["lift_before_sd"], -3) == 26_000
    # "narrows the uncertainty about the lift only from about 26,000 to about 23,000"
    assert round(h["lift_after_sd"], -3) == 23_000
    # "a return of 1.27x on the 77,000 withheld"
    assert round(h["test_roas_before"]["mean"], 2) == 1.27
    # annual ROAS is about 1.2 times the test return, between about 1.0 and 1.5
    assert rounded(h["annual_over_test_roas_before"], 1) == (1.2, 1.0, 1.5)


def test_chapter36_profit_lens(r):
    final = r["ch36"]["final"]
    profit = r["ch36"]["profit_m"]
    printed = {
        # channel: (contribution margin, profit after media, ROI per cent)
        "search": (1.55, 0.15, 11),
        "meta": (1.24, -0.26, -17),
        "youtube": (0.88, 0.08, 10),
        "tv": (1.80, 0.40, 28),
        "ooh": (0.74, -0.16, -17),
    }
    for ch, (margin, after, roi) in printed.items():
        assert round(MARGIN * final[ch]["contribution_m"], 2) == margin, ch
        assert round(profit[ch], 2) == after, ch
        assert round(100 * profit[ch] / final[ch]["spend_m"]) == roi, ch
    assert round(MARGIN * final["total"]["contribution_m"], 2) == 6.21
    assert round(profit["total"], 2) == 0.21
    assert round(100 * profit["total"] / final["total"]["spend_m"]) == 4


# ---- Chapter 37 and Appendix F: marginal returns ---------------------------------


def test_marginal_return_intervals(r):
    c = r["ch37"]
    printed = {
        "search": (1.39, 1.02, 1.78),
        "meta": (1.11, 0.64, 1.58),
        "youtube": (1.87, 1.04, 2.74),
        "tv": (2.22, 2.11, 2.33),
        "ooh": (1.39, 1.03, 1.78),
    }
    for ch, value in printed.items():
        assert rounded(c[ch]["marginal"]) == value, ch
        assert rounded(c[ch]["average"]) == CALIBRATED_PRINTED[ch], ch
    truth = {"search": 1.40, "meta": 0.90, "youtube": 1.80, "tv": 2.10, "ooh": 1.47}
    for ch, value in truth.items():
        assert round(c[ch]["true_marginal"], 2) == value, ch
    # four of the five intervals contain the truth; Television's lower end sits above it
    inside = [
        ch
        for ch in CHANNELS
        if c[ch]["marginal"]["lo"] <= c[ch]["true_marginal"] <= c[ch]["marginal"]["hi"]
    ]
    assert sorted(inside) == ["meta", "ooh", "search", "youtube"]
    assert c["tv"]["marginal"]["lo"] > c["tv"]["true_marginal"]
    # Meta's next pound is the one the model overstates most
    over = {ch: c[ch]["marginal"]["mean"] - c[ch]["true_marginal"] for ch in CHANNELS}
    assert max(over, key=over.get) == "meta"
    # the ranking at the margin (Television first, YouTube second, Meta last) matches the truth
    fitted = sorted(CHANNELS, key=lambda ch: -c[ch]["marginal"]["mean"])
    true = sorted(CHANNELS, key=lambda ch: -c[ch]["true_marginal"])
    assert fitted[:2] == true[:2] == ["tv", "youtube"]
    assert fitted[-1] == true[-1] == "meta"


def test_chapter37_break_even_probabilities(r):
    c = r["ch37"]
    p = {ch: c[ch]["p_marginal_above_break_even"] for ch in CHANNELS}
    assert round(p["search"] * 100) == 1  # "1%"
    assert round(p["ooh"] * 100) == 1  # "1%"
    assert 0 < p["meta"] < 0.005  # "under 1%"
    assert round(p["youtube"] * 100) == 39  # "39%", "about four draws out of ten"
    assert p["tv"] > 0.99  # "over 99%"


# ---- Chapter 38 ------------------------------------------------------------------


def plan_row(plan_m):
    return {ch: round(v, 2) for ch, v in plan_m.items()}


def test_chapter38_naive_plan(r):
    c = r["ch38"]
    naive = c["plans"]["naive_40"]
    assert plan_row(naive["plan_m"]) == {
        "search": 1.29,
        "meta": 0.90,
        "youtube": 1.12,
        "tv": 1.96,
        "ooh": 0.73,
    }
    assert round(naive["gain_m"]["mean"], 2) == 0.58
    assert round(naive["gain_m"]["mean"], 3) == 0.582  # Appendix F: 582,000
    # "from about 70,000 to 1.1 million"
    assert round(naive["gain_m"]["lo"], 2) == 0.07
    assert round(naive["gain_m"]["hi"], 1) == 1.1
    # "the 320,000 added to YouTube"
    assert round(naive["plan_m"]["youtube"] - 0.80, 2) == 0.32
    # YouTube's next pound across the curve draws: about 1 at the low end
    assert round(c["youtube_marginal"]["lo"]) == 1
    assert round(c["youtube_marginal"]["hi"], 2) == 2.75


def test_chapter38_optimiser_and_operational_plan(r):
    c = r["ch38"]
    own = c["plans"]["bounded_optimiser"]
    assert plan_row(own["plan_m"]) == {
        "search": 1.40,
        "meta": 1.20,
        "youtube": 0.85,
        "tv": 1.65,
        "ooh": 0.90,
    }
    assert round(own["gain_m"]["mean"], 3) == 0.277
    op = c["plans"]["operational"]
    assert plan_row(op["plan_m"]) == {
        "search": 1.30,
        "meta": 1.25,
        "youtube": 0.85,
        "tv": 1.65,
        "ooh": 0.95,
    }
    assert round(op["gain_m"]["mean"], 3) == 0.266
    assert round(op["profit_m"], 3) == 0.133
    # "11,000 less than the optimiser's own plan"
    assert round(own["gain_m"]["mean"] - op["gain_m"]["mean"], 3) == 0.011
    assert (round(op["gain_m"]["lo"], 2), round(op["gain_m"]["hi"], 2)) == (0.11, 0.44)
    assert op["p_gain"] > 0.99
    # "in every one, television's next pound earns more than Meta's"
    assert c["p_tv_marginal_above_meta"] == 1.0
    # Search and OOH have the same marginal return, about 1.4
    m = r["ch37"]
    assert (
        round(m["search"]["marginal_from_curves"], 1)
        == round(m["ooh"]["marginal_from_curves"], 1)
        == 1.4
    )


def test_chapter38_fixed_budget_arithmetic(r):
    s = r["ch38"]["fixed_budget_steps"]
    assert round(s["youtube"]["revenue"], -3) == 92_000
    assert round(s["ooh"]["revenue"], -3) == 69_000
    assert round(s["meta"]["revenue"], -3) == 61_000
    # Meta near the new budget about 1.2x, YouTube about 1.8x
    assert round(s["meta"]["per_pound"], 1) == 1.2
    assert round(s["youtube"]["per_pound"], 1) == 1.8
    # moving 50,000 from Meta to YouTube earns about 30,000 more revenue
    assert round(s["youtube"]["revenue"] - s["meta"]["revenue"], -4) == 30_000
    # contribution margin on the two increases: about 46,000 and 35,000
    assert round(MARGIN * s["youtube"]["revenue"], -3) == 46_000
    assert round(MARGIN * s["ooh"]["revenue"], -3) == 35_000
    # returning OOH's 50,000 would raise short-run profit by about 15,000
    assert round(-s["ooh"]["profit"], -3) == 15_000
    assert s["youtube"]["profit"] < 0


def test_chapter38_plans_table(r):
    p = r["ch38"]["plans"]
    printed = {
        # plan: (gain, profit, chance per cent)
        "ch27_unconstrained": (0.81, 0.40, 94),
        "ch27_constrained": (0.45, 0.22, 99),
        "naive_40": (0.58, 0.29, 97),
        "operational": (0.27, 0.13, None),
    }
    for plan, (gain, profit, chance) in printed.items():
        assert round(p[plan]["gain_m"]["mean"], 2) == gain, plan
        assert round(p[plan]["profit_m"], 2) == profit, plan
        if chance is not None:
            assert round(p[plan]["p_gain"] * 100) == chance, plan
    assert p["operational"]["p_gain"] > 0.99  # "over 99%"
    rows = {
        "ch27_unconstrained": {
            "search": 1.19,
            "meta": 0.68,
            "youtube": 1.13,
            "tv": 3.00,
            "ooh": 0.00,
        },
        "ch27_constrained": {
            "search": 1.22,
            "meta": 1.20,
            "youtube": 1.04,
            "tv": 1.82,
            "ooh": 0.72,
        },
    }
    for plan, row in rows.items():
        assert plan_row(p[plan]["plan_m"]) == row, plan


def test_chapter38_scenarios_table(r):
    c = r["ch38"]
    s = c["scenarios"]
    assert round(s["cut15"]["revenue_m"]["mean"], 1) == 11.2
    assert round(c["operational_revenue_m"]["mean"], 1) == 12.7
    assert round(s["up10"]["revenue_m"]["mean"], 1) == 13.7
    assert round(s["cut15"]["marginal_roas"], 1) == 1.7
    assert round(s["up10"]["marginal_roas"], 1) == 1.6
    # profit rises in every posterior draw under the cut and falls in every one under the rise
    assert s["cut15"]["p_profit_up"] == 1.0
    assert s["up10"]["p_profit_up"] == 0.0
    # what the 2.07x average would have suggested
    roas = r["ch36"]["final"]["total"]["roas"]
    assert round(roas * 0.9, 2) == 1.86
    assert round(roas * 0.6, 2) == 1.24
    # the long-term effect that would cancel each cut's advantage
    cut, targeted = s["cut15"], s["targeted_cut"]
    assert round(2 * cut["profit_change_m"], 2) == 0.28
    assert round(5 * 2 * cut["profit_change_m"] / -cut["difference_m"]["mean"]) == 1  # a fifth
    assert (
        round(2 * 2 * targeted["profit_change_m"] / -targeted["difference_m"]["mean"]) == 1
    )  # half


# ---- Chapter 39 ------------------------------------------------------------------


def test_chapter39_value_of_information_in_text(r):
    c = r["ch39"]
    meta = c["voi"]["meta"]
    assert (round(meta["marginal_p10"], 2), round(meta["marginal_p90"], 2)) == (0.76, 1.49)
    assert round(meta["moved_m"], 2) == 0.54
    assert round(c["voi"]["youtube"]["moved_m"], 2) == 0.35
    assert round(c["voi"]["search"]["moved_m"], 2) == 0.32
    assert round(c["voi"]["ooh"]["moved_m"], 2) == 0.32
    # Meta at its floor in the pessimistic draws, up by a sixth in the optimistic ones,
    # with YouTube cut to make room
    assert round(meta["low_plan_m"]["meta"], 2) == 1.20
    assert round(6 * (meta["high_plan_m"]["meta"] / 1.50 - 1)) == 1
    assert meta["high_plan_m"]["youtube"] < 0.80
    assert round(c["half_year_gain_m"], 3) == 0.133
