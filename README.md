# Marketing Mix Modelling - Companion Code

Companion code and notebooks for the book *Marketing Mix Modelling* by Mark Stent.

[![CI](https://github.com/markstent/mmm-book-companion/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/markstent/mmm-book-companion/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/markstent/mmm-book-companion?color=informational)](https://github.com/markstent/mmm-book-companion/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyMC 6.2](https://img.shields.io/badge/PyMC-6.2-1f6fb2.svg)](https://www.pymc.io/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Notebooks tested](https://img.shields.io/badge/notebooks-28%20tested%20in%20CI-orange.svg?logo=jupyter&logoColor=white)](#running-the-tests)
[![Citation](https://img.shields.io/badge/cite-CITATION.cff-blueviolet.svg)](CITATION.cff)

## What this is

This repository is the companion code for *Marketing Mix Modelling* by Mark Stent.
It contains the synthetic dataset, the small helper library, and every notebook referenced in the book.
It is for readers who want to run the book's models themselves, inspect the underlying code, and experiment beyond what is printed on the page.
The notebooks build marketing mix models from first principles in raw PyMC, so every prior and every transform is written out where you can see it.
Once the fundamentals are in place, a single bridge notebook shows how the same ideas map onto PyMC-Marketing.

This repository contains code only.
The book's text and figures are not included here.

The synthetic Sedge Shoes data has known true effects built in, and every estimate the book's case study (Part IX) prints is what the companion's case-study model reports when fitted to that data, so a run of the code lands on the numbers the book prints, within the sampling noise described in [Reproducing the book's numbers](#reproducing-the-books-numbers).
There are two exceptions, which the chapters label as illustrative: the six-month refresh in Chapter 39 and the long-term television effect Chapter 38 borrows from Chapter 29 are not computed by the code.
Every notebook is committed with its outputs, so you can read the results on GitHub before you run anything.

## Quickstart

```bash
uv sync
uv run jupyter lab
```

`uv sync` installs Python 3.12 and all dependencies (including JupyterLab) into a local virtual environment.
`uv run jupyter lab` launches JupyterLab with that environment, so you can open any notebook under `notebooks/` or `exercises/`.
Once set up, `make test` runs the full test suite, including every notebook.

PyMC compiles its models with your system C compiler.
If compilation fails with a linker error such as "must link with libSystem.dylib" on a macOS beta, point `SDKROOT` at an installed stable SDK, for example `SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk uv run pytest`.

## Repository layout

```
companion/
├── src/mmm_book/            # Installable helper package
│   ├── sedge.py             # Synthetic "Sedge Shoes" data generator
│   ├── adstock.py           # Geometric adstock transform
│   ├── saturation.py        # Hill saturation transform
│   ├── reference.py         # Reference aggregation, features, controls, and model
│   ├── case_model.py        # The Chapter 35 to 39 regional case-study model
│   ├── case_results.py      # Every number Part IX prints, from the fitted models
│   ├── optimise.py          # Per-draw response curves and a bounded greedy budget allocator
│   ├── experiment.py        # The Chapter 36 Meta geo holdout numbers
│   ├── metrics.py           # ROAS and contribution-share helpers
│   └── plotting.py          # Shared plotting style
├── notebooks/
│   ├── labs/                 # 11 method labs, one per modelling chapter
│   ├── case_study/            # 6 case-study notebooks
│   └── bridge_pymc_marketing.ipynb   # Raw PyMC -> PyMC-Marketing bridge
├── exercises/                # 5 reader exercises
│   └── solutions/             # Matching worked solutions
├── examples/                 # 11 short runnable scripts, one per code-heavy topic
├── results/part9.json        # The Part IX numbers, written by case_results (the book's figures read it)
├── tests/                    # Unit tests, including tests that pin the book's printed numbers
├── Makefile                  # sync, unit, notebooks, test, lint
└── pyproject.toml
```

## Chapter-to-notebook map

### Labs

| Chapter | Title | Notebook |
| --- | --- | --- |
| 13 | Carryover, Lag and Adstock | [notebooks/labs/13_adstock_and_carryover.ipynb](notebooks/labs/13_adstock_and_carryover.ipynb) |
| 14 | Saturation and Diminishing Returns | [notebooks/labs/14_saturation.ipynb](notebooks/labs/14_saturation.ipynb) |
| 15 | Putting the Model Together | [notebooks/labs/15_building_the_model.ipynb](notebooks/labs/15_building_the_model.ipynb) |
| 17 | Choosing and Testing Priors | [notebooks/labs/17_priors_and_prior_predictive.ipynb](notebooks/labs/17_priors_and_prior_predictive.ipynb) |
| 18 | Fitting the Model | [notebooks/labs/18_fitting_and_diagnostics.ipynb](notebooks/labs/18_fitting_and_diagnostics.ipynb) |
| 19 | Hierarchical and Time-Varying MMM | [notebooks/labs/19_hierarchical_and_time_varying.ipynb](notebooks/labs/19_hierarchical_and_time_varying.ipynb) |
| 21 | Prediction and Posterior Predictive Checks | [notebooks/labs/21_posterior_predictive_checks.ipynb](notebooks/labs/21_posterior_predictive_checks.ipynb) |
| 23 | Synthetic Recovery and Identifiability Testing | [notebooks/labs/23_synthetic_recovery.ipynb](notebooks/labs/23_synthetic_recovery.ipynb) |
| 24 | Experiments and Calibration | [notebooks/labs/24_experiments_and_calibration.ipynb](notebooks/labs/24_experiments_and_calibration.ipynb) |
| 26 | Response Curves and Marginal Returns | [notebooks/labs/26_response_curves.ipynb](notebooks/labs/26_response_curves.ipynb) |
| 27 | Budget Optimisation | [notebooks/labs/27_budget_optimisation.ipynb](notebooks/labs/27_budget_optimisation.ipynb) |

### Case study

| Chapter | Title | Notebook |
| --- | --- | --- |
| 34 | The Brief and the Data | [notebooks/case_study/34_the_brief_and_the_data.ipynb](notebooks/case_study/34_the_brief_and_the_data.ipynb) |
| 35 | A First Model, and Why It Failed | [notebooks/case_study/35_a_first_model_and_why_it_failed.ipynb](notebooks/case_study/35_a_first_model_and_why_it_failed.ipynb) |
| 36 | Validation and Calibration | [notebooks/case_study/36_validation_and_calibration.ipynb](notebooks/case_study/36_validation_and_calibration.ipynb) |
| 37 | From Average Return to the Next Pound | [notebooks/case_study/37_average_to_next_pound.ipynb](notebooks/case_study/37_average_to_next_pound.ipynb) |
| 38 | Optimising the Budget | [notebooks/case_study/38_optimising_the_budget.ipynb](notebooks/case_study/38_optimising_the_budget.ipynb) |
| 39 | The Recommendation, and What Happened Next | [notebooks/case_study/39_the_recommendation.ipynb](notebooks/case_study/39_the_recommendation.ipynb) |

### Bridge notebook and exercises

[notebooks/bridge_pymc_marketing.ipynb](notebooks/bridge_pymc_marketing.ipynb) is not tied to a single chapter.
It re-fits the national Sedge model the labs use with PyMC-Marketing, so that once you understand what the library does for you, you can compare it with the from-scratch PyMC version.
The `exercises/` directory holds 5 standalone exercises for readers who want extra practice, each with a worked solution in `exercises/solutions/`.

### Example scripts

`examples/` holds 11 short scripts, one per code-heavy topic (adstock, saturation, building, priors, fitting, hierarchy, posterior predictive checks, synthetic recovery, calibration, response curves and budget optimisation).
Each runs on its own in seconds, for example `uv run python examples/ex_adstock.py`, and prints a short result.
The PyMC ones use a small, fast sampler configuration on a two-region, one-year slice of Sedge, so they show the pattern rather than publication-quality posteriors.
[examples/README.md](examples/README.md) describes each script, and `tests/test_examples.py` runs every one of them in CI.

## The synthetic Sedge Shoes dataset

Most of the labs run against a synthetic dataset called Sedge Shoes: a made-up multi-region weekly retail business with five marketing channels (TV, search, Meta, YouTube, and OOH).
Revenue in each region-week is built from a baseline (trend, seasonality and product launches), scaled up or down by price and promotion depth (non-media treatments that Sedge sets itself) and by the controls (distribution, stock, weather and category demand), plus a contribution from each channel.
Each channel's spend is adstocked and saturated, then scaled so that its aggregate ROAS matches a known ground-truth value exactly, by construction.
Because the true ROAS is known, Sedge is a useful test bed: later labs can check whether a fitted model actually recovers it.

The table below gives the generator's built-in truth.
These are the values the book's teaching chapters 25 to 28 use, and the ones Appendix F compares the fitted model with.

| | Book | Generator default |
| --- | --- | --- |
| Regions, weeks, rows | 18, 156, 2,808 | 18, 156, 2,808 |
| Annual revenue | about 48m pounds | about 48m pounds |
| Annual media | 6.0m (TV 1.4, Search 1.4, Meta 1.5, YouTube 0.8, OOH 0.9) | within about 5 per cent of each |
| Average ROAS | TV 2.4, Search 2.7, Meta 1.75, YouTube 2.0, OOH 1.5 | exactly these (`GROUND_TRUTH`) |
| Marginal ROAS at current spend | TV 2.1, Search 1.4, Meta 0.9, YouTube 1.8, OOH 1.47 | within about 5 per cent (`true_marginal_roas()`) |

`tests/test_book_ledger.py` asserts every row of that table, so the generator cannot drift away from the book without a test failing.

You can regenerate the dataset, or generate your own variant, directly from the package:

```python
from mmm_book.sedge import (
    GROUND_TRUTH,
    generate_sedge,
    ground_truth_contributions,
    true_marginal_roas,
)

df = generate_sedge()  # 18 regions, 156 weeks, seed 42
contributions = ground_truth_contributions()

print(GROUND_TRUTH)
# {'tv': 2.4, 'search': 2.7, 'meta': 1.75, 'youtube': 2.0, 'ooh': 1.5}
print(true_marginal_roas())
```

`GROUND_TRUTH` holds the exact by-construction average ROAS for every channel.
`ground_truth_contributions` returns each channel's true revenue contribution, row-aligned with the DataFrame from `generate_sedge` for the same arguments and seed.
Generation is fully deterministic for a given seed, so the same call always reproduces the same dataset.
Television is stored as GRPs; `TV_COST_PER_GRP` (400 pounds) converts them to a spend equivalent so that television's ROAS can be stated on the same footing as the other channels.

## How the code relates to the book

The code re-creates the book's analysis for teaching.
Where it differs from the book's model, the difference is deliberate, and each one is listed here so you are not surprised.

- **Adstock convention.** The book's Chapter 13 table (100,000 pounds, then 50,000, 25,000, 12,500) uses the plain recursion `A_t = x_t + alpha * A_(t-1)`, which is `geometric_adstock(..., normalize=False)`.
  The labs and the generator use `normalize=True`, which rescales the lag weights to sum to one and truncates at `l_max = 12`.
  The two differ by a constant factor, so the shape of the carryover is identical.
  PyMC-Marketing's `GeometricAdstock` component normalises by default as well, although its low-level function does not.
  Lab 13 and `examples/ex_adstock.py` show both.
- **Transform hyperparameters are fixed in the labs.** The labs' reference model uses the generator's true adstock alpha, Hill slope and half-saturation point for each channel, so only the channel scales, baseline and controls are estimated.
  Fixing them keeps the labs quick and focused on one idea at a time.
  On real data you would not know these values, and the labs say so.
  The case-study model estimates the transforms.
- **A simplified model.** The reference model is national rather than regional, and Lab 15 builds it in Chapter 15's stages: a baseline (intercept, trend, one Fourier pair for seasonality and a single launch indicator), promotion depth as a non-media treatment, and price, distribution, stock, weather and category controls, then the media.
  Its betas are non-negative, each on a feature that averages one, so a beta is a channel's average weekly contribution in millions of pounds rather than the book's contribution at full saturation.
  Its priors are the ones Lab 17 chooses with prior predictive checks, set on revenue's own scale.
- **Sampler settings.** Every lab that samples uses four chains of 1,000 draws after 1,000 warm-up steps with a target acceptance rate of 0.95 (0.99 for the time-varying fits in Lab 19 and for the PyMC-Marketing bridge).
  Every sampled lab prints its divergences and largest `r_hat` for each main fit, and none has a divergence or an `r_hat` above 1.01.
  Labs 13, 14 and 17 run no MCMC.
  Lab 18 also shows, on purpose, the warnings a shorter, less cautious run produces, and how to work through them.
  Every interval the labs report is a 90 per cent interval, as in the book.
- **The case-study model.** `mmm_book.case_model` is the model the book describes for Chapters 35 to 39, fitted to the regional panel.
  It estimates adstock retention and Hill half-saturation for every channel, puts priors directly on ROAS, pools regional media effects, and projects the Meta holdout through the fitted curves.
  Each fit runs four chains of 2,000 draws (8,000 in all) after 1,000 warm-up steps at a target acceptance rate of 0.97 and is cached in `.cache/`.
- **Part IX reports the fitted model.** `mmm_book.case_results` turns the five case-study fits into every number Part IX prints, and writes them to `results/part9.json`, which the book's Part IX figures read.
  Response curves and the budget optimiser use `case_model.posterior_curves`: one curve per posterior draw, built from that draw's own adstock, saturation and regional scales, so the uncertainty in the curves' shapes, not only their heights, reaches every plan and every probability.
  `tests/test_book_part9.py` pins the printed values.
- **The teaching chapters use the built-in truth.** Chapters 27 and 28 do their optimisation arithmetic on `optimise.ResponseCurves` with `ledger_betas`, the curves a model would have if it had recovered the generator's true values exactly; `tests/test_book_optimiser.py` pins those figures.
- **Held-out validation without leakage.** With `train_weeks` set, `prepare_case_data` scales the controls and trend, and sets each region's reference activity and the spend that defines ROAS, from the training weeks alone (Chapter 21).
  Lab 21 runs the same kind of forward split on the reference model (`reference_model(nat, n_train=...)`) and checks the coverage of its 90 per cent intervals on the held-out weeks.
- **The Meta geo holdout.** Chapter 36's experiment pauses Meta for six weeks in eight treatment regions, with eight matched controls.
  A holdout measures the return on the spend that was switched off, which is an average-type quantity for that slice; a push would measure the return on the extra spend, which is a marginal-type quantity.
  `mmm_book.experiment` recomputes the design: about 77,000 pounds withheld (12,900 a week), an expected lost revenue on the true curve of 120,000 pounds over the six weeks plus a two-week cooldown, and a standard error of about 45,000 pounds from placebo windows.
  The observed result is 138,000 pounds of lift, a ROAS of 1.79 plus or minus 0.58.
  The generated panel records business-as-usual Meta spend in the test weeks, so the experiment is information the panel does not contain.
  It enters the case-study model once, as a likelihood term on the lift the model's own curve implies for the same regions and weeks.
  `examples/ex_calibration.py` uses a simpler stand-in on the national model (ROAS times withheld spend).
- **An experiment enters once.** Sedge's two television tests set the television prior and are not added again as likelihood terms; the Meta holdout is kept out of every prior.
- **Lab 24 and Exercise 4** calibrate Search to a simulated experiment, a generic teaching example that does not use the Sedge experiment.
  The experiment's standard error is widened for transport to Chapter 24's $\sigma_{\text{eff}}$ before it enters the likelihood.

## Running the tests

```bash
make test
```

This runs the unit tests in `tests/` and executes every notebook under `notebooks/` and `exercises/` end to end, using the same commands as CI.
You can also run the pieces separately with `make unit`, `make notebooks`, and `make lint`.

### Reproducing the book's numbers

The figures Part IX prints are the reference run, saved in `results/part9.json` by `mmm_book.case_results`.
Most of `tests/test_book_part9.py` checks the book's printed values against that file, so those tests give the same answer on every machine and take seconds.
Three further tests refit the five case-study models on your machine (several minutes on a laptop the first time; the fits are then cached in `.cache/`).
A refit is a fresh Monte Carlo run: on a different operating system or processor, small differences in floating-point arithmetic send the sampler's chains down different paths.
Posterior means typically land within about 0.02 of the reference run, interval ends and plan gains within about 0.07, and pound figures within a few per cent.
The tests check that a refit stays within that noise, passes every sampler diagnostic (no divergences, R-hat at most 1.01) and reaches the book's main conclusions: calibration moves Meta up, television's next pound beats Meta's, the plans beat the current mix, and the ranking of experiments in Chapter 39 holds.
The case-study notebooks are committed with the outputs of the reference run, so what you read on GitHub matches the book exactly.
The GitHub Actions workflow in `.github/workflows/ci.yml` runs this same suite on every push and pull request.

To regenerate the outputs saved in the notebooks after a change, run for each notebook:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/labs/13_adstock_and_carryover.ipynb
```

## Exercises and solutions

Each file in `exercises/` is a standalone notebook designed to be attempted before looking at the answer.
The matching solution for each exercise lives in `exercises/solutions/`, under the same filename.
Solutions are also executed as part of the test suite, so they are guaranteed to run against the current version of the package.

## Contributing

Issues and pull requests are welcome, particularly for bug fixes, clearer explanations in the code, and additional test coverage.
Please keep the modelling code in `notebooks/labs/` and `notebooks/case_study/` in first-principles PyMC, since that style is a deliberate teaching choice tied to the book.
If you are proposing a larger change, please open an issue first to discuss it.

## Citation

If you use this code or the Sedge Shoes dataset generator, please cite it.
GitHub's "Cite this repository" button reads [`CITATION.cff`](CITATION.cff); the BibTeX equivalent is:

```bibtex
@software{stent_mmm_companion,
  author  = {Stent, Mark},
  title   = {Marketing Mix Modelling - Companion Code},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/markstent/mmm-book-companion}
}
```

## License

The code in this repository is licensed under the MIT License; see [LICENSE](LICENSE) for details.
The text and figures of *Marketing Mix Modelling* are not included in this repository and remain copyright Mark Stent.
