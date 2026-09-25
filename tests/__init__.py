"""Lightweight, laptop-safe smoke tests run on synthetic data (see each script's docstring).

These are NOT unit tests of individual functions in isolation; they exercise
the full pipeline (normalization -> blocking -> retrieval -> features ->
model -> thresholding -> output -> official validator) end to end on a tiny
synthetic dataset, specifically so that logic bugs are caught without ever
needing to run the heavy full-scale (multi-million-row) real data through
the pipeline on a resource-constrained machine.
"""
