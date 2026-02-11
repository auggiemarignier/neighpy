from .search import NASearcher
from .appraise import NAAppraiser

try:
    from ._version import __version__
except ImportError:
    import warnings
    warnings.warn(
        "Could not import version information. Using fallback version 0.0.0. "
        "This may indicate an installation issue.",
        ImportWarning,
    )
    __version__ = "0.0.0"
