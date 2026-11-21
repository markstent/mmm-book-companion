"""Greedy numpy-only budget allocation across channels (no scipy)."""

# [book:budget_imports-start]
from mmm_book.optimise import ResponseCurves, band, greedy_allocate, ledger_betas
from mmm_book.reference import national_aggregate
from mmm_book.sedge import CHANNELS, generate_sedge

# [book:budget_imports-end]

# [book:budget-start]
nat = national_aggregate(generate_sedge())
# Betas that reproduce the book's average ROAS; use fitted betas,
# idata.posterior["beta_<channel>"].mean(), in a real analysis.
curves = ResponseCurves(nat, ledger_betas(nat))

# No channel may move by more than 20 per cent of its current spend.
lower, upper = band(curves, 0.20)
alloc = greedy_allocate(curves, lower, upper)

print(f"{'channel':<10}{'current':>12}{'optimised':>12}{'change':>9}")
for ch in CHANNELS:
    now, new = curves.current_money[ch], alloc[ch]
    print(f"{ch:<10}{now:>12,.0f}{new:>12,.0f}{new / now - 1:>9.1%}")
gain = curves.total_contribution(alloc) - curves.total_contribution(curves.current_money)
print(f"expected gain: {gain * 52:,.0f} GBP a year at the same total budget")
# [book:budget-end]
