"""Greedy numpy-only budget allocation across channels (no scipy)."""

# [book:budget_imports-start]
from mmm_book.optimise import (
    ResponseCurves,
    band,
    greedy_allocate,
    ledger_betas,
)
from mmm_book.reference import national_aggregate
from mmm_book.sedge import CHANNELS, generate_sedge

# [book:budget_imports-end]

# [book:budget-start]
nat = national_aggregate(generate_sedge())
# Betas that reproduce the book's average ROAS; in a real
# analysis use fitted betas, e.g.
# idata.posterior["beta_<channel>"].mean().
curves = ResponseCurves(nat, ledger_betas(nat))

# No channel may move by more than 20 per cent of its spend.
lower, upper = band(curves, 0.20)
alloc = greedy_allocate(curves, lower, upper)

print(f"{'channel':<10}{'current':>12}", end="")
print(f"{'optimised':>12}{'change':>9}")
for ch in CHANNELS:
    now, new = curves.current_money[ch], alloc[ch]
    change = new / now - 1
    print(f"{ch:<10}{now:>12,.0f}{new:>12,.0f}{change:>9.1%}")
before = curves.total_contribution(curves.current_money)
gain = curves.total_contribution(alloc) - before
note = "at the same total budget"
print(f"expected gain: {gain * 52:,.0f} GBP a year {note}")
# [book:budget-end]
