"""Canonical neutral entry point for the frozen-artifact evidence audit.

Current UQ eligibility is gated by the accepted likelihood-/budget-matched
artifact. Legacy UQ and proxy outputs remain readable but non-propagating.
"""

from __future__ import annotations

from run_reviewer_resolution_audit import *  # noqa: F401,F403
from run_reviewer_resolution_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
