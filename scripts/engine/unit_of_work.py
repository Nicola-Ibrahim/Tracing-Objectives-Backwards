from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .system import get_root_dir
from .ui import Logger


class WorkSkipped(Exception):
    """Internal exception to gracefully skip a unit of work block."""

    pass


class Command:
    """
    Encapsulates a shell command as an 'Action Object'.
    Execution happens immediately upon instantiation if 'cmd' is provided.
    """

    def __init__(
        self,
        ctx: UnitOfWork | Logger,
        cmd: str | None = None,
        description: str | None = None,
        cwd: str | Path | None = None,
        interactive: bool = False,
        stream: bool = False,
        confirm: bool = False,
        skip_confirm: bool = False,
        exit_on_error: bool = False,
        **options: Any,
    ):
        # Context extraction
        if isinstance(ctx, UnitOfWork):
            self.logger = ctx.logger
            self.uow = ctx
        else:
            self.logger = ctx
            self.uow = None

        self.cmd = cmd
        self.description = description
        self.cwd = cwd or get_root_dir()
        self.interactive = interactive
        self.stream = stream
        self.confirm = confirm
        self.skip_confirm = skip_confirm
        self.exit_on_error = exit_on_error
        self.options = options
        self.success = False

        if self.cmd:
            self.success = self._execute()

        # Register result with UnitOfWork if present
        if self.uow:
            self.uow.register_result(self)

    def _execute(self) -> bool:
        """Internal execution engine."""
        if self.description:
            # Handle step-level confirmation
            if self.confirm and not self.skip_confirm:
                if not self.logger.confirm(f"Run step: {self.description}?"):
                    self.logger.info_step("Skipped by user.")
                    return False

            self.logger.step(self.description)

        try:
            if self.interactive:
                result = subprocess.run(self.cmd, shell=True, cwd=self.cwd)
                returncode = result.returncode
            elif self.stream:
                process = subprocess.Popen(
                    self.cmd,
                    shell=True,
                    cwd=self.cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                for line in process.stdout:
                    if line.strip():
                        self.logger.output(line)

                process.wait()
                returncode = process.returncode
            else:
                result = subprocess.run(
                    self.cmd,
                    shell=True,
                    cwd=self.cwd,
                    capture_output=True,
                    text=True,
                )
                if result.stdout and result.stdout.strip():
                    self.logger.output(result.stdout)

                if result.returncode != 0 and result.stderr:
                    self.logger.error("Raw error output:")
                    self.logger.error_output(result.stderr)

                returncode = result.returncode

            if returncode == 0:
                if self.description:
                    self.logger.success(f"Successfully finished: {self.description}")
                return True
            else:
                if self.description:
                    self.logger.error(f"Failed to complete: {self.description}")

                if self.exit_on_error:
                    sys.exit(returncode)
                return False

        except Exception as e:
            if self.description:
                self.logger.error(f"Critical exception during: {self.description}")
            self.logger.info_step(f"Exception: {str(e)}")
            if self.exit_on_error:
                sys.exit(1)
            return False


class UnitOfWork:
    """
    Acts as a 'Unit of Work' providing a visual and behavioral context.
    Each UnitOfWork represents an atomic, isolated block of execution.
    """

    def __init__(
        self,
        title: str,
        info: str | None = None,
    ):
        self.title = title
        self.info = info
        self.logger = Logger()
        self.results: list[Command] = []
        self._start_time: float | None = None

    def register_result(self, cmd: Command) -> None:
        """Capture the result of a command run within this unit."""
        self.results.append(cmd)

    def __enter__(self) -> "UnitOfWork":
        self._start_time = time.time()

        # 1. Display Header
        self.logger.header(self.title)

        # 2. Display Context/Info
        if self.info:
            self.logger.info(self.info)

        # 3. Handle Confirmation logic has been moved to Command level
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self._start_time if self._start_time else 0

        # Don't show summary if work was skipped
        if exc_type is WorkSkipped:
            return True

        # Aggregate Result
        if self.results:
            success_count = sum(1 for r in self.results if r.success)
            total_count = len(self.results)
            all_ok = success_count == total_count

            status_msg = f"{self.title} completed"
            detail_msg = f"({success_count}/{total_count} actions, {duration:.1f}s)"

            if all_ok:
                self.logger.success(f"{status_msg} {detail_msg}")
            else:
                self.logger.error(f"{status_msg} with failures {detail_msg}")

        return False
