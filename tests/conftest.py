"""Hypothesis profiles, selected with HYPOTHESIS_PROFILE (default: dev).

dev: quick local runs; ci: more examples in CI; thorough: run before opening a pull request.
Physics examples can be slow, so no per-example deadline.
"""

import os

from hypothesis import settings

settings.register_profile("dev", max_examples=100, deadline=None)
settings.register_profile("ci", max_examples=200, deadline=None, print_blob=True)
settings.register_profile("thorough", max_examples=2000, deadline=None, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
