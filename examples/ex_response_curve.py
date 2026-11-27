"""Build a response curve for one channel from a point beta."""

# [book:response_curve_imports-start]
import numpy as np

from mmm_book.optimise import (
    ResponseCurves,
    ledger_betas,
    to_money,
)
from mmm_book.reference import national_aggregate
from mmm_book.sedge import generate_sedge

# [book:response_curve_imports-end]

# [book:response_curve-start]
CHANNEL = "search"

nat = national_aggregate(generate_sedge())
# A fitted model supplies the betas in a real analysis, e.g.
# idata.posterior["beta_search"].mean(). Here they reproduce
# the book's average ROAS, giving Chapter 23's teaching curve.
curves = ResponseCurves(nat, ledger_betas(nat))

now = curves.current_native[CHANNEL]
for level in np.linspace(0, now * 2, 5):
    spend = to_money(CHANNEL, level)
    revenue = curves.contribution(CHANNEL, level)
    print(f"weekly spend={spend:9,.0f}", end="  ")
    print(f"contribution={revenue:10,.0f}")

average = curves.average_return(CHANNEL)
marginal = curves.marginal_return(CHANNEL)
print(f"average return at current spend:  {average:.2f}")
print(f"marginal return at current spend: {marginal:.2f}")
# [book:response_curve-end]
