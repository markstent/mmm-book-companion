# Examples

These short scripts show the modelling pattern behind each code-heavy chapter.
The book prints short listings taken from these files, so they are CI-tested.
Each listing sits between a pair of marker comments, `# [book:NAME-start]` and `# [book:NAME-end]`; leave the markers in place when you edit a script.

Every script is self-contained, imports from the installed `mmm_book` package, runs to completion in a few seconds and prints a short result.
`tests/test_examples.py` runs every `examples/ex_*.py` script as a subprocess and asserts that it exits with status 0, so a script cannot drift out of date without CI failing.

Any script that samples with PyMC uses a small, fast configuration (`draws=200, tune=200, chains=2, cores=1, target_accept=0.9`) with the shared `RANDOM_SEED`, so the run is deterministic and quick.
That is enough to show the modelling pattern the chapter describes, though not to produce publication-quality posteriors.

## Scripts

- `ex_adstock.py` - geometric adstock (carryover) applied to a single burst of spend, in the book's raw convention and the normalised convention the labs use.
- `ex_saturation.py` - Hill saturation response values across a range of spend levels.
- `ex_build_model.py` - build the reference national-aggregate MMM without sampling it, and print its structure.
- `ex_priors.py` - draw a prior predictive sample from the reference model.
- `ex_fit.py` - fit the reference model and print `az.summary` of the channel betas.
- `ex_hierarchical.py` - a minimal partial-pooling model with per-region betas pooled to a channel mean.
- `ex_ppc.py` - fit the reference model and run a posterior predictive check against observed revenue.
- `ex_synthetic_recovery.py` - recover each channel's ROAS from a fit and compare it to the known ground truth. The recovery is approximate by design, because the national model is fitted to data generated region by region, and that gap sets up the calibration example below.
- `ex_calibration.py` - add an experiment likelihood term for the Meta channel (a simple stand-in for the book's Chapter 32 geo holdout) and refit.
- `ex_response_curve.py` - build a response curve for one channel from a point beta and read off average and marginal return.
- `ex_budget_optimisation.py` - greedy numpy-only budget reallocation across channels (no scipy).
- `ex_power.py` - Chapter 2: simulate a four-week, 20,000-a-week campaign against 20,000 of weekly noise and count how often it is detected (about half the time).
- `ex_confounding.py` - Chapter 5: Table 5.2's visitors example, and a promotion riding on television with and without the promotion as a control.
- `ex_data_audit.py` - Chapter 11: run the audits on `generate_sedge(faults=True)` and print each planted fault found.
- `ex_scenarios.py` - Chapter 25: four costings of the 15 per cent budget cut on the teaching curves.
- `ex_drift.py` - Chapter 29: the run-of-four rule on a predictive interval, with a fast stand-in for the model's predictive draws.
- `ex_broken_extract.py` - Chapter 30: drop 30 per cent of Search spend from the extract and watch ROAS rise by 43 per cent.
- `ex_readout.py` - Chapter 34: write Sedge's one-page readout as Markdown from `results/part9.json`.

## Running

```bash
uv run python examples/ex_adstock.py
```

Or run all of them as tests:

```bash
uv run pytest tests/test_examples.py -v
```

## Keeping these runnable

If you change an interface in `src/mmm_book`, update the examples that use it in the same change.
Keep each script short so it can be read at a glance.
`tests/test_examples.py` will fail if any script stops running, so a broken example cannot merge.
