"""
server/app.py — DebateFloor server entry point.

This module is the deployment boundary. All business logic lives in app/.
External clients and evaluation scripts interact via HTTP (/reset, /step, /state)
and should never import app/ internals directly.
"""
import uvicorn
from app.main import app  # noqa: F401 — re-exported for uvicorn discovery

__all__ = ["app"]


def serve(host: str = "0.0.0.0", port: int = 7860, workers: int = 1) -> None:
    """Start the DebateFloor environment server."""
    uvicorn.run("server.app:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    serve()
