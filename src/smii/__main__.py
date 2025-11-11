"""Command line entry point for SeaMeInIt demos."""

from __future__ import annotations

from typing import Sequence

from .app import build_cli

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    return build_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
