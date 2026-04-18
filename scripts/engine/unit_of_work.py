from __future__ import annotations

import subprocess
import sys
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
        logger: Logger,
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
        self.logger = logger
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

    def __enter__(self) -> "UnitOfWork":
        # 1. Display Header
        self.logger.header(self.title)

        # 2. Display Context/Info
        if self.info:
            self.logger.info(self.info)

        # 3. Handle Confirmation logic has been moved to Command level
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is WorkSkipped:
            return True
        return False
