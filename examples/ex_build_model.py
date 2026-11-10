"""Build the reference national-aggregate MMM, without sampling it."""

from mmm_book.reference import national_aggregate, reference_model
from mmm_book.sedge import generate_sedge

df = generate_sedge(n_weeks=52, n_regions=2)
nat = national_aggregate(df)
model = reference_model(nat)

print(model.str_repr())
print("free parameters:", [rv.name for rv in model.free_RVs])
print("weeks of national data:", len(nat))
