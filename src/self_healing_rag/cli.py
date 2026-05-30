import typer

from self_healing_rag.graph import ask
from self_healing_rag.index import build_local_index

app = typer.Typer(help="Self-healing RAG command line interface.")


@app.command()
def answer(question: str) -> None:
    """Ask the self-healing RAG graph a question."""
    typer.echo(ask(question))


@app.command()
def ingest(
    input_path: str = typer.Argument(..., help="File or directory containing source documents."),
    output_path: str = typer.Option(
        "data/index/chunks.jsonl",
        "--output",
        "-o",
        help="Path where the local chunk index should be written.",
    ),
) -> None:
    """Build a local chunk index from source documents."""
    chunks = build_local_index(input_path=input_path, output_path=output_path)
    typer.echo(f"Indexed {len(chunks)} chunks into {output_path}")


if __name__ == "__main__":
    app()
