"""Core package for SeaMeInIt pipelines and utilities."""

# Register the optional AVIF codec before modules query Pillow feature support.
try:  # pragma: no cover - depends on the optional visual dependency
    import pillow_avif  # noqa: F401
except ImportError:  # pragma: no cover - JPEG/PNG-only installations remain valid
    pass

__all__ = ["pipelines"]
