"""Geometric adstock applied to a single burst of spend, in both conventions."""

# [book:adstock_imports-start]
import numpy as np

from mmm_book.adstock import geometric_adstock

# [book:adstock_imports-end]

# [book:adstock-start]
spend = np.array([100.0, 0.0, 0.0, 0.0, 0.0])

# Chapter 13's table: A_t = x_t + alpha * A_(t-1), unscaled.
book = geometric_adstock(spend, alpha=0.5, l_max=5, normalize=False)
# Labs and PyMC-Marketing: weights rescaled to sum to one.
normalised = geometric_adstock(spend, alpha=0.5, l_max=5, normalize=True)

print("spend:                     ", spend)
print("adstocked (book, raw):     ", np.round(book, 3))
print("adstocked (normalised):    ", np.round(normalised, 3))
print("ratio normalised / book:   ", np.round(normalised / book, 4), "(constant)")
# [book:adstock-end]
