from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from protocol_risk_analyzer.backend.analysis_service import run_protocol_analysis

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def analyze(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output_dir: Path = typer.Option(Path("out"), "--output-dir", "-o"),
    model: str | None = typer.Option(None, "--model"),
    chunk_chars: int = typer.Option(14_000, "--chunk-chars"),
    overlap_chars: int = typer.Option(1_200, "--overlap-chars"),
) -> None:
    """
    Analyze a clinical protocol document and output risk reports.
    """
    chosen_model, stats, output_paths = run_protocol_analysis(
        input_path=input_path,
        output_dir=output_dir,
        model=model,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
    )
    console.print(
        "[cyan]"
        f"Prepared {stats.chunk_count} chunks from {stats.analyzed_pages}/{stats.total_pages} "
        f"pages using model {chosen_model}."
        "[/cyan]"
    )

    console.print("[green]Analysis complete.[/green]")
    console.print(f"- JSON: {output_paths.analysis_json_path}")
    console.print(f"- Markdown: {output_paths.report_md_path}")
    console.print(f"- PDF: {output_paths.report_pdf_path}")


if __name__ == "__main__":
    app()
