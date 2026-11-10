"""Build a response curve for one channel from a point beta."""

import numpy as np

from mmm_book.optimise import ResponseCurves, ledger_betas, to_money
from mmm_book.reference import national_aggregate
from mmm_book.sedge import generate_sedge

CHANNEL = "search"

nat = national_aggregate(generate_sedge())
# In a real analysis the betas come from a fitted model, e.g.
# idata.posterior["beta_search"].mean(). Here we use the betas that reproduce
# the book's average ROAS ledger, so the curve is the teaching curve Chapters 27 and 28 use.
curves = ResponseCurves(nat, ledger_betas(nat))

now = curves.current_native[CHANNEL]
for level in np.linspace(0, now * 2, 5):
    spend, revenue = to_money(CHANNEL, level), curves.contribution(CHANNEL, level)
    print(f"weekly spend={spend:9,.0f}  contribution={revenue:10,.0f}")

print(f"average return at current spend:  {curves.average_return(CHANNEL):.2f}")
print(f"marginal return at current spend: {curves.marginal_return(CHANNEL):.2f}")
