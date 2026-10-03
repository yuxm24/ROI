"""Low-level runtime utilities.

Kept intentionally import-light: training modules import concrete utilities
from their defining modules so startup does not acquire optional dependencies.
"""
