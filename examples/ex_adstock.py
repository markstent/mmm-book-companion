"""Geometric adstock applied to a single burst of spend, in both conventions."""

import numpy as np

from mmm_book.adstock import geometric_adstock

spend = np.array([100.0, 0.0, 0.0, 0.0, 0.0])

# The book's Chapter 13 table: A_t = x_t + alpha * A_(t-1), no rescaling.
book = geometric_adstock(spend, alpha=0.5, l_max=5, normalize=False)
# The convention used by the labs and by PyMC-Marketing's component:
# lag weights rescaled to sum to one.
normalised = geometric_adstock(spend, alpha=0.5, l_max=5, normalize=True)

print("spend:                     ", spend)
print("adstocked (book, raw):     ", np.round(book, 3))
print("adstocked (normalised):    ", np.round(normalised, 3))
print("ratio normalised / book:   ", np.round(normalised / book, 4), "(constant)")
