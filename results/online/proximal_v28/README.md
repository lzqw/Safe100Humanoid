# Proximal v28: smoother CBF response exploration

v28 tested whether a smaller exponential-CBF alpha could replace late, large
toe-clearance corrections with an earlier and smoother response. The test used
one fresh 128-pair development seed at the fixed 18 cm riser, slope 0.8, and
the original 15 cm recovery window.

No candidate met the unchanged gate's 80% filter-on success threshold:

- alpha 4: 65.625%
- alpha 6: 74.21875%
- alpha 8: 74.21875%
- alpha 10: 78.125%

The common filter-off rate was 64.0625%, with 91.3043% toe-riser alignment.
Alpha 10 remained best, so reducing alpha did not solve filter-induced
regressions. v28 stopped at development; no formal calibration or training was
started.
