# Examples

These short scripts show the modelling pattern behind each code-heavy chapter.
The book itself prints no code listings, so these files are its runnable counterpart.

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
- `ex_calibration.py` - add an experiment likelihood term for the Meta channel (a simple stand-in for the book's Chapter 36 geo holdout) and refit.
- `ex_response_curve.py` - build a response curve for one channel from a point beta and read off average and marginal return.
- `ex_budget_optimisation.py` - greedy numpy-only budget reallocation across channels (no scipy).

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
