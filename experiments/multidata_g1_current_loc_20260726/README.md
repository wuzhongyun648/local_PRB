# Six-Dataset g1-Current LocPRB

This isolated orchestration reuses the production dataset loaders, graph
managers, EE-Net training schedule, and scratch Numba LocPRB implementation.
It runs formal horizons with seeds 200–209 and `evaluation_every=0`, so fixed
evaluation construction cannot advance the global NumPy RNG used by EE-Net
training.

The experiment covers MovieLens, AmazonFashion, Facebook, Collab, PPA, and
Vessel. Four restart-safe queues are balanced using observed formal-run times.
Each GPU queue runs 10 concurrent worker lanes. Every lane is pinned to one
distinct CPU core and all numerical-library thread counts are fixed at one.
Thus the four-GPU launch runs up to 40 experiment processes concurrently.

This protocol is a cross-dataset g1-current-style comparison. It is not an
event-by-event replay of the earlier all-method runs, which used seeds 0–9 and
periodic evaluation.
