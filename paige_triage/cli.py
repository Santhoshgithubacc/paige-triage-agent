"""CLI: run a Paige-style deep investigation against the demo incident."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from paige_triage.runtime import build_demo_supervisor

app = typer.Typer(add_completion=False, help="Paige-style SRE triage agent")
console = Console()


@app.command()
def investigate(
    steer: Optional[str] = typer.Option(
        None, help="Mid-run user steering, e.g. 'also check DNS'"
    ),
    checkpoint_dir: Path = typer.Option(Path(".paige_checkpoints")),
    timeout: float = typer.Option(60.0, help="Seconds to wait for completion"),
) -> None:
    """Fan out hypotheses, gather evidence, synthesize recommendations."""

    async def _run() -> None:
        def on_progress(msg: str) -> None:
            console.print(f"[cyan]•[/cyan] {msg}")

        supervisor, bundle = build_demo_supervisor(
            checkpoint_dir=checkpoint_dir, on_progress=on_progress
        )
        console.print(
            Panel.fit(
                f"[bold]{bundle['incident'].title}[/bold]\n"
                f"service={bundle['incident'].service.name} id={bundle['incident'].id}",
                title="Paige Triage",
            )
        )
        await supervisor.start(bundle["incident"], bundle["alerts"])
        if steer:
            # Give sub-agents a moment to start, then inject (priority 0)
            await asyncio.sleep(0.2)
            await supervisor.inject_user_input(steer)
        state = await supervisor.wait_until_complete(timeout=timeout)

        table = Table(title="Hypotheses")
        table.add_column("Status")
        table.add_column("Confidence")
        table.add_column("Statement")
        for h in state.hypotheses:
            table.add_row(h.status.value, f"{h.confidence:.2f}", h.statement)
        console.print(table)

        console.print(Panel(state.synthesis or "(none)", title="Synthesis"))
        for rec in state.recommendations:
            console.print(f"[green]→[/green] {rec}")
        await supervisor.shutdown()

    asyncio.run(_run())


@app.command("api")
def run_api(host: str = "127.0.0.1", port: int = 8088) -> None:
    """Start the FastAPI control plane."""
    import uvicorn

    uvicorn.run("paige_triage.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
