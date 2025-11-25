from pathlib import Path
import json
import typer
from embedding_pipe import build_vector_store
from retrieval_pipe import get_character_info

app = typer.Typer(help="LangChain Mistral character info extractor")


@app.command("compute-embeddings")
def compute_embeddings(
    books_dir: str = typer.Option(
        "./stories",
        "--books-dir",
        "-b",
        help="Directory containing .txt story files.",
    ),
    db_path: str = typer.Option(
        "./db",
        "--db-path",
        "-d",
        help="Directory to load the Chroma vector store.",
    ),
):
    try: 
        build_vector_store(books_dir, db_path)
        typer.echo("[INFO] Vector store built and persisted successfully.")
    except FileNotFoundError as e:
        typer.echo(f"[ERROR] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"[ERROR] Unexpected error: {e}")
        raise typer.Exit(code=1)


@app.command("get-character-info")
def get_character_info_cmd(
    name: str = typer.Argument(..., help="Character name to look up."),
    db_path: str = typer.Option(
        "./db",
        "--db-path",
        "-d",
        help="Directory where the Chroma vector store is persisted.",
    ),
):
    try:
        result = get_character_info(name, db_path)
    except FileNotFoundError as e:
        typer.echo(f"[ERROR] {e}")
        raise typer.Exit(code=1)  
    except Exception as e:
        typer.echo(f"[ERROR] Unexpected error: {e}")
        raise typer.Exit(code=1)

  
    try:
        parsed = json.loads(result)
        typer.echo(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        typer.echo(result)


if __name__ == "__main__":
    app()
