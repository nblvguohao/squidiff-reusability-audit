"""Typer CLI for the biomedical reusability gate.

Commands:
    reuse-gate inventory   Record environment and hardware inventory.
    reuse-gate data        Download and verify datasets.
    reuse-gate gate        Evaluate candidate hard gates.
    reuse-gate decide      Run deterministic candidate selection.
    reuse-gate report      Generate feasibility reports.
    reuse-gate run-selected  Execute the selected candidate's Tier workflow.
"""

from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def inventory() -> None:
    """Record environment and hardware inventory."""
    typer.echo("inventory: not yet implemented")


@app.command()
def data() -> None:
    """Download and verify datasets."""
    typer.echo("data: not yet implemented")


@app.command()
def gate() -> None:
    """Evaluate candidate hard gates."""
    typer.echo("gate: not yet implemented")


@app.command()
def decide() -> None:
    """Run deterministic candidate selection."""
    typer.echo("decide: not yet implemented")


@app.command()
def report() -> None:
    """Generate feasibility reports."""
    typer.echo("report: not yet implemented")


@app.command()
def run_selected() -> None:
    """Execute the selected candidate's Tier workflow."""
    typer.echo("run-selected: not yet implemented")


if __name__ == "__main__":
    app()
