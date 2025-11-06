# ruff: noqa: E402
import os

if not (
    (disable_truststore := os.getenv("DISABLE_TRUSTSTORE"))
    and disable_truststore.lower() == "true"
):
    import truststore  # noqa: F401

    truststore.inject_into_ssl()  # noqa: F401

import asyncio
import contextlib
import http.server
import json
import logging.config
import os
import pathlib
import signal
from contextlib import asynccontextmanager
from typing import cast

import structlog

from langgraph_runtime.lifespan import lifespan
from langgraph_runtime.metrics import get_metrics

logger = structlog.stdlib.get_logger(__name__)


async def health_and_metrics_server():
    port = int(os.getenv("PORT", "8080"))
    ok = json.dumps({"status": "ok"}).encode()
    ok_len = str(len(ok))

    class HealthAndMetricsHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # Skip logging for /ok and /metrics endpoints
            if getattr(self, "path", None) in ["/ok", "/metrics"]:
                return
            # Log other requests normally
            super().log_message(format, *args)

        def do_GET(self):
            path = getattr(self, "path", None)
            if path == "/ok":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", ok_len)
                self.end_headers()
                self.wfile.write(ok)
            elif path == "/metrics":
                metrics = get_metrics()
                worker_metrics = cast(dict[str, int], metrics["workers"])
                workers_max = worker_metrics["max"]
                workers_active = worker_metrics["active"]
                workers_available = worker_metrics["available"]

                project_id = os.getenv("LANGSMITH_HOST_PROJECT_ID")
                revision_id = os.getenv("LANGSMITH_HOST_REVISION_ID")

                metrics = [
                    "# HELP lg_api_workers_max The maximum number of workers available.",
                    "# TYPE lg_api_workers_max gauge",
                    f'lg_api_workers_max{{project_id="{project_id}", revision_id="{revision_id}"}} {workers_max}',
                    "# HELP lg_api_workers_active The number of currently active workers.",
                    "# TYPE lg_api_workers_active gauge",
                    f'lg_api_workers_active{{project_id="{project_id}", revision_id="{revision_id}"}} {workers_active}',
                    "# HELP lg_api_workers_available The number of available (idle) workers.",
                    "# TYPE lg_api_workers_available gauge",
                    f'lg_api_workers_available{{project_id="{project_id}", revision_id="{revision_id}"}} {workers_available}',
                ]

                metrics_response = "\n".join(metrics).encode()
                metrics_len = str(len(metrics_response))

                self.send_response(200)
                self.send_header(
                    "Content-Type", "text/plain; version=0.0.4; charset=utf-8"
                )
                self.send_header("Content-Length", metrics_len)
                self.end_headers()
                self.wfile.write(metrics_response)
            else:
                self.send_error(http.HTTPStatus.NOT_FOUND)

    with http.server.ThreadingHTTPServer(
        ("0.0.0.0", port), HealthAndMetricsHandler
    ) as httpd:
        logger.info(f"Health and metrics server started at http://0.0.0.0:{port}")
        try:
            await asyncio.to_thread(httpd.serve_forever)
        finally:
            httpd.shutdown()


async def entrypoint(
    grpc_port: int | None = None, entrypoint_name: str = "python-queue"
):
    from langgraph_api import logging as lg_logging
    from langgraph_api.api import user_router

    lg_logging.set_logging_context({"entrypoint": entrypoint_name})
    tasks: set[asyncio.Task] = set()
    tasks.add(asyncio.create_task(health_and_metrics_server()))

    original_lifespan = user_router.router.lifespan_context if user_router else None

    @asynccontextmanager
    async def combined_lifespan(
        app, with_cron_scheduler=False, grpc_port=None, taskset=None
    ):
        async with lifespan(
            app,
            with_cron_scheduler=with_cron_scheduler,
            grpc_port=grpc_port,
            taskset=taskset,
        ):
            if original_lifespan:
                async with original_lifespan(app):
                    yield
            else:
                yield

    async with combined_lifespan(
        None, with_cron_scheduler=False, grpc_port=grpc_port, taskset=tasks
    ):
        await asyncio.gather(*tasks)


async def main(grpc_port: int | None = None, entrypoint_name: str = "python-queue"):
    """Run the queue entrypoint and shut down gracefully on SIGTERM/SIGINT."""
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _handle_signal() -> None:
        logger.warning("Received termination signal, initiating graceful shutdown")
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGTERM, _handle_signal)
    except (NotImplementedError, RuntimeError):
        signal.signal(signal.SIGTERM, lambda *_: _handle_signal())

    entry_task = asyncio.create_task(
        entrypoint(grpc_port=grpc_port, entrypoint_name=entrypoint_name)
    )
    await stop_event.wait()

    logger.warning("Cancelling queue entrypoint task")
    entry_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await entry_task


if __name__ == "__main__":
    from langgraph_api import config

    config.IS_QUEUE_ENTRYPOINT = True
    with open(pathlib.Path(__file__).parent.parent / "logging.json") as file:
        loaded_config = json.load(file)
        logging.config.dictConfig(loaded_config)
    try:
        import uvloop  # type: ignore[unresolved-import]

        uvloop.install()
    except ImportError:
        pass
    # run the entrypoint
    asyncio.run(main())
