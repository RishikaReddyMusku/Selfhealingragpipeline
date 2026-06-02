import typer

from self_healing_rag.config import settings
from self_healing_rag.evaluation import run_evaluation
from self_healing_rag.graph import ask
from self_healing_rag.index import build_indexes

app = typer.Typer(help="Self-healing RAG command line interface.")


@app.command()
def answer(
    question: str,
    index_path: str = typer.Option(
        settings.local_index_path,
        "--index",
        "-i",
        help="Path to the local chunk index.",
    ),
) -> None:
    """Ask the self-healing RAG graph a question."""
    typer.echo(ask(question, index_path=index_path))


@app.command()
def ingest(
    input_path: str = typer.Argument(..., help="File or directory containing source documents."),
    output_path: str = typer.Option(
        settings.local_index_path,
        "--output",
        "-o",
        help="Path where the local chunk index should be written.",
    ),
    vector: bool = typer.Option(
        True,
        "--vector/--no-vector",
        help="Also build the Chroma vector index.",
    ),
) -> None:
    """Build a local chunk index from source documents."""
    chunks = build_indexes(input_path=input_path, output_path=output_path, build_vector=vector)
    typer.echo(f"Indexed {len(chunks)} chunks into {output_path}")


@app.command()
def eval(
    questions_path: str = typer.Option(
        "data/demo_questions.json",
        "--questions",
        "-q",
        help="Path to a JSON list of evaluation questions.",
    ),
    index_path: str = typer.Option(
        settings.local_index_path,
        "--index",
        "-i",
        help="Path to the local chunk index.",
    ),
) -> None:
    """Run a lightweight RAG quality evaluation."""
    summary = run_evaluation(questions_path=questions_path, index_path=index_path)
    typer.echo(f"Questions: {summary['total']}")
    typer.echo(f"Approval rate: {summary['approval_rate']:.0%}")
    typer.echo(f"Citation rate: {summary['citation_rate']:.0%}")
    typer.echo(f"Fallback rate: {summary['fallback_rate']:.0%}")
    typer.echo(f"Average attempts: {summary['average_attempts']:.2f}")


if __name__ == "__main__":
    app()
