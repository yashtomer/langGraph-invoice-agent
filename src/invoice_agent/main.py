"""Process entry point — boots logging, DB, scheduler, and the FastAPI server."""
from __future__ import annotations

import signal
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .config import get_settings
from .db import init_db
from .logging_setup import configure_logging, get_logger
from .scheduler import build_scheduler
from .webhook.legal import build_legal_router
from .webhook.server import build_router

log = get_logger(__name__)


def create_app() -> FastAPI:
    configure_logging()
    s = get_settings()
    init_db(s)

    sched = build_scheduler(s)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        sched.start()
        log.info("app.started", port=s.webhook_port, tz=s.timezone)
        try:
            yield
        finally:
            sched.shutdown(wait=False)
            log.info("app.stopped")

    app = FastAPI(title="Invoice Agent", version="0.1.0", lifespan=lifespan)
    app.include_router(build_router(s))
    app.include_router(build_legal_router())
    return app


def run() -> None:
    s = get_settings()
    configure_logging()

    def _handle_sigterm(*_):
        log.info("signal.sigterm")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    uvicorn.run(
        "invoice_agent.main:create_app",
        host=s.webhook_host,
        port=s.webhook_port,
        factory=True,
        reload=False,
    )


if __name__ == "__main__":
    run()
