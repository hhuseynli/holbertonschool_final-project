"""Real-data adapters for Azerbaijani listing platforms.

Each adapter turns a platform's public pages into rows of the canonical
schema in :mod:`bakuml.data.schema`, so the rest of the pipeline does not
know or care which site a listing came from. Cross-platform duplicates are
removed once, downstream, by :mod:`bakuml.data.dedup`.
"""

from bakuml.data.sources.base import HarvestResult, polite_get

__all__ = ["HarvestResult", "polite_get"]
