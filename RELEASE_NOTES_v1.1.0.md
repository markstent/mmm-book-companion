# v1.1.0 (draft)

This release follows the book's renumbering of its chapters.
Nothing about the models, the data or the printed numbers changes.

## Notebooks renamed

Notebooks follow the book's new chapter numbers.
Where the book merged two chapters, each notebook keeps a letter suffix (`a`, `b`).
Labs 13, 14, 15, 17, 18 and 19 keep their names.

| Old path | New path |
| --- | --- |
| `notebooks/labs/21_posterior_predictive_checks.ipynb` | `notebooks/labs/20_posterior_predictive_checks.ipynb` |
| `notebooks/labs/23_synthetic_recovery.ipynb` | `notebooks/labs/21_synthetic_recovery.ipynb` |
| `notebooks/labs/24_experiments_and_calibration.ipynb` | `notebooks/labs/22_experiments_and_calibration.ipynb` |
| `notebooks/labs/26_response_curves.ipynb` | `notebooks/labs/23_response_curves.ipynb` |
| `notebooks/labs/27_budget_optimisation.ipynb` | `notebooks/labs/24_budget_optimisation.ipynb` |
| `notebooks/case_study/34_the_brief_and_the_data.ipynb` | `notebooks/case_study/31a_the_brief_and_the_data.ipynb` |
| `notebooks/case_study/35_a_first_model_and_why_it_failed.ipynb` | `notebooks/case_study/31b_a_first_model_and_why_it_failed.ipynb` |
| `notebooks/case_study/36_validation_and_calibration.ipynb` | `notebooks/case_study/32_validation_and_calibration.ipynb` |
| `notebooks/case_study/37_average_to_next_pound.ipynb` | `notebooks/case_study/33a_average_to_next_pound.ipynb` |
| `notebooks/case_study/38_optimising_the_budget.ipynb` | `notebooks/case_study/33b_optimising_the_budget.ipynb` |
| `notebooks/case_study/39_the_recommendation.ipynb` | `notebooks/case_study/34_the_recommendation.ipynb` |

Chapter, lab and appendix references in docstrings, comments, tests, notebooks and the README now use the new numbering.
The statistics and marketing primers are cited as Primer 2 and Primer 1.

## `results/part9.json` keys

The top-level keys are now stable names instead of chapter numbers.
The values are unchanged.

| Old key | New key |
| --- | --- |
| `ch27` | `teaching_optimisation` |
| `ch34` | `brief` |
| `ch35` | `first_model` |
| `ch36` | `validation` |
| `ch37` | `marginal` |
| `ch38` | `optimisation` |
| `ch39` | `recommendation` |

Inside `optimisation`, the `plans` keys `ch27_unconstrained` and `ch27_constrained` are now `teaching_unconstrained` and `teaching_constrained`.
`mmm_book.case_results` writes the new keys.
The `mmm_book.case_results` functions that compute each block take the same names: `chapter27` is now `teaching_optimisation`, `chapter34` to `chapter39` are now `brief`, `first_model`, `validation`, `marginal`, `optimisation` and `recommendation`, and `CH27_PLANS` is now `TEACHING_PLANS`.

## Listing markers

The book prints short listings from the CI-tested scripts.
Each `examples/ex_*.py` now has one pair of marker comments, `# [book:NAME-start]` and `# [book:NAME-end]`, around the code the book prints.
`src/mmm_book/case_model.py` has `case-model` markers around the media, likelihood and noise block of the case-study model.
`src/mmm_book/reference.py` has `reference-model` markers around the model block of the national reference model.
Markers are comments and change no behaviour.
