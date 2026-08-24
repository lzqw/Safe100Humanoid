"""v29 entry point for deterministic fixed-riser CBF evaluation.

The mature v26 evaluator already records the required per-episode action,
intervention, kick, return, and completion fields.  v29 deliberately reuses
that implementation without changing historical result files, while extending
the fixed-riser configurator to the nominal 13 cm D0 height.
"""

from evaluate_cbf_teacher_v26 import main


if __name__ == "__main__":
    main()
