from typing import Any

from ..application.diagnose_engines import (
    RunDiagnosticsCommand,
    RunDiagnosticsService,
)


async def run_diagnostics_task(
    ctx: dict,
    task_id: str,
    command_data: dict[str, Any],
):
    """
    ARQ task for the asynchronous RunDiagnosticsService.
    Dependencies are retrieved from the worker context (ctx).
    """
    service: RunDiagnosticsService = ctx["run_diagnostics_service"]

    command = RunDiagnosticsCommand(**command_data)
    await service.execute(command, task_id=task_id)
