"""Shared console helpers for the CLI."""

from __future__ import annotations

import sys
from typing import NoReturn

import typer
from rich.console import Console

console = Console()
error_console = Console(stderr=True)


def fail(message: str, *, hint: str | None = None, code: int = 1) -> NoReturn:
    """Print an error (and optional hint) to stderr and exit."""
    error_console.print(f"[bold red]error[/bold red] {message}")
    if hint:
        error_console.print(f"[dim]hint:[/dim] {hint}")
    raise typer.Exit(code)


def is_tty() -> bool:
    return sys.stdout.isatty()
