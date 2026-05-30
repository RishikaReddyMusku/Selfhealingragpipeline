import typer

from self_healing_rag.graph import ask

app = typer.Typer(help="Self-healing RAG command line interface.")


@app.command()
def answer(question: str) -> None:
    """Ask the self-healing RAG graph a question."""
    typer.echo(ask(question))


if __name__ == "__main__":
    app()

