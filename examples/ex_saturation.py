"""Hill saturation curve values across a range of spend levels."""

import numpy as np

from mmm_book.saturation import hill_saturation

spend = np.linspace(0, 200, 5)
response = hill_saturation(spend, half=100.0, slope=1.5)

for s, r in zip(spend, response, strict=True):
    print(f"spend={s:6.1f}  response={r:.3f}")
