import click
import uvicorn


@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind the server to.")
@click.option("--port", default=8000, help="Port to bind the server to.")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload.")
def main(host, port, reload):
    """
    Launch the FastAPI web server for Tracing Objectives Backwards.
    """
    uvicorn.run("src.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
