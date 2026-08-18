# Proximal v27: short recovery window

v27 prospectively fixed the v26 sloped x-z toe-clearance barrier at a uniform
18 cm riser, slope 0.8, and a shortened 2 cm post-edge recovery window.

The development-only selection sample improved paired base-policy success from
57.8125% to 81.25%. The independent 256-pair confirmatory calibration did not
replicate that result:

- filter off: 157/256 (61.328125%)
- filter on: 189/256 (73.828125%)
- paired rescues: 72/99 base failures (72.7273%)
- paired regressions: 40
- toe-riser alignment coverage: 64/99 (64.6465%)

The frozen gate required at least 80% filter-on success, at least 80% alignment
coverage, and at least 60% rescue. v27 therefore terminated before training.
No adapted policy or final evaluation was produced.

The prospective protocol, development summary, execution marker, exact gate
counts, and terminal calibration summary are included in this directory.
