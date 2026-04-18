from __future__ import annotations
from .ui import logger as default_logger

class JobSkipped(Exception):
    """Internal exception to gracefully skip a job block."""
    pass

class Job:
    """
    Encapsulates a logical unit of work (a "Job") with its own
    visual lifecycle, metadata, and optional user confirmation.
    """
    def __init__(
        self, 
        title: str, 
        info: str | None = None, 
        confirm: bool = False, 
        logger = None
    ):
        self.title = title
        self.info = info
        self.confirm = confirm
        self.logger = logger or default_logger

    def __enter__(self):
        # 1. Display Header
        self.logger.header(self.title)
        
        # 2. Display Context/Info
        if self.info:
            self.logger.info(self.info)
        
        # 3. Handle Confirmation
        if self.confirm:
            if not self.logger.confirm(f"Run step: {self.title}?"):
                this_is_not_really_an_error = "Skipped by user."
                self.logger.info_step(this_is_not_really_an_error)
                raise JobSkipped()
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Gracefully handle the skip signal
        if exc_type is JobSkipped:
            return True
        
        # Propagate all other exceptions (including Critical ones)
        return False
