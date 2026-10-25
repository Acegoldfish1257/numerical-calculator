# Staged implementation checkpoints

Each numerical method was implemented with its own tests and checked before proceeding to the next method. Topic pages were then exercised through Streamlit AppTest.

1. Shared validation, restricted expression parsing, consistent results, bisection and GUI: 28 initial checks passed.
2. Remaining root finders and comparison page: 39 checks passed.
3. Lagrange, Hermite, interpolation bounds, fitting and splines: method checks and all added pages passed.
4. Five quadrature rules, finite differences and RK4: complete suite reached 79 passing checks.
5. Arithmetic, Taylor, matrix methods, best approximation, optimization and eigenpairs: complete suite reached 126 passing checks.
6. Broader malformed-input testing found that comparison-page shared expression errors were only shown in rows. Fixed with page-level expression validation. A browser check found a clipped metric on a narrow viewport; shortened its headline and added a full-precision caption.
7. Added a reproducible 38-case expected/actual table and final validation: **211 tests passed**, including valid and invalid input flows on all 39 Streamlit pages. All 38 verification cases passed, and `pip check` reported no broken requirements. A live browser check confirmed the table and Matplotlib plot; `app_preview.png` records the inspected output.

These counts are historical checkpoints, not a substitute for running the current tests.
