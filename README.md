# A Tight Theory of Error Feedback Algorithms in Distributed Optimization

Repository for "A Tight Theory of Error Feedback Algorithms in Distributed Optimization", accepted as a poster at ICML 2026.

This repository contains the code to reproduce the figures currently wired into `experiments.py`.

## Running Experiments

To reproduce all available experiment figures:
```bash
python experiments.py all
```

To reproduce specific results:
```bash
python experiments.py "Figure 1"
python experiments.py "Figure 2"
python experiments.py "Figure 3"
```

## Certificates and Verification

The `certificates/` directory contains the supplementary verification artifacts used for the paper's theoretical and empirical claims:

- `certificates/EF21/`: numerical and symbolic checks for the EF21 result. The notebook `theorem1.ipynb` documents the numerical certificate construction, and the WolframScript files `ef21_multiworker.wls` and `ef21_multiworker_poscheck.wls` contain symbolic certificate and positivity checks.
- `certificates/EF/`: analogous artifacts for the EF result, with `theorem2.ipynb` and the WolframScript checks `ef_multiworker.wls` and `ef_multiworker_random_poscheck.wls`.
- `certificates/empirical_laws/`: notebooks for the empirical laws in Section 4, covering the step-size rule, Lyapunov structure, n=2 rate law, and EControl tuning. Shared notebook utilities are in `notebook_setup.py`.
