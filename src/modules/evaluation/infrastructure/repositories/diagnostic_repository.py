from pathlib import Path
from typing import Any

from ....shared.config import ROOT_PATH
from ....shared.infrastructure.processing.files.json import JsonFileHandler
from ...domain.aggregates.diagnostic_report import DiagnosticReport
from ...domain.interfaces.base_diagnostic_repository import BaseDiagnosticRepository


class RunNumberManager:
    """
    Determines the next sequential evaluation run number for a model version directory.
    Format: diagnostics/inverse/dataset/mdn/v1/run{N}-YYYY-MM-DD/
    """

    def get_next_run_number(self, version_directory: Path) -> int:
        existing_runs = []
        if version_directory.exists():
            for entry in version_directory.iterdir():
                if entry.is_dir() and entry.name.startswith("run"):
                    try:
                        # Extract N from runN-...
                        parts = entry.name.split("-")
                        run_str = parts[0].replace("run", "")
                        run_num = int(run_str)
                        existing_runs.append(run_num)
                    except (ValueError, IndexError):
                        continue

        return max(existing_runs) + 1 if existing_runs else 1


class FileSystemDiagnosticRepository(BaseDiagnosticRepository):
    """
    Persists DiagnosticReport entities as JSON using sequential run numbering.
    Location: ROOT/diagnostics/<dataset>/<direction>/<type>/v<version>/run<N>-<date>/
    """

    def __init__(self, base_path: str = "diagnostics"):
        self._base_storage_path = ROOT_PATH / base_path
        self._json_handler = JsonFileHandler()
        self._run_manager = RunNumberManager()
        self._base_storage_path.mkdir(parents=True, exist_ok=True)

    def _compute_version_directory(
        self,
        mapping_direction: str,
        dataset_name: str,
        estimator_type: str,
        version: int,
    ) -> Path:
        return (
            self._base_storage_path
            / dataset_name
            / mapping_direction
            / estimator_type
            / f"v{version}"
        )

    def _find_run_dir(self, version_dir: Path, run_number: int) -> Path | None:
        if not version_dir.exists():
            return None
        prefix = f"run{run_number}-"
        for entry in version_dir.iterdir():
            if entry.is_dir() and entry.name.startswith(prefix):
                return entry
        return None

    def save(self, report: DiagnosticReport) -> int:
        version_dir = self._compute_version_directory(
            report.engine.mapping_direction,
            report.dataset_name,
            report.engine.type,
            report.engine.version,
        )
        version_dir.mkdir(parents=True, exist_ok=True)

        # Assign sequential run number
        next_run = self._run_manager.get_next_run_number(version_dir)
        report.run_number = next_run

        # runN-YYYY-MM-DD
        run_name = f"run{next_run}-{report.created_at.strftime('%Y-%m-%d')}"
        run_dir = version_dir / run_name
        run_dir.mkdir(exist_ok=True)

        save_path = run_dir / "evaluation.json"

        # Pydantic dict() (or model_dump in newer pydantic) handles nested models
        self._json_handler.save(report.model_dump(), save_path)

        return next_run

    def load(
        self,
        engine_type: str,
        engine_version: int,
        run_number: int,
        dataset_name: str,
        mapping_direction: str = "inverse",
    ) -> DiagnosticReport:
        version_dir = self._compute_version_directory(
            mapping_direction, dataset_name, engine_type, engine_version
        )
        run_dir = self._find_run_dir(version_dir, run_number)

        if not run_dir or not (run_dir / "evaluation.json").exists():
            raise FileNotFoundError(
                f"Evaluation run {run_number} not found for {engine_type} v{engine_version}."
            )

        data = self._json_handler.load(run_dir / "evaluation.json")

        # Use factory method for clean reconstruction
        return DiagnosticReport.from_data(data)

    def get_all_runs(
        self,
        engine_type: str,
        engine_version: int,
        dataset_name: str,
        mapping_direction: str = "inverse",
    ) -> list[DiagnosticReport]:
        version_dir = self._compute_version_directory(
            mapping_direction, dataset_name, engine_type, engine_version
        )
        if not version_dir.exists():
            return []

        results = []
        for entry in version_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("run"):
                try:
                    run_num = int(entry.name.split("-")[0].replace("run", ""))
                    results.append(
                        self.load(
                            engine_type,
                            engine_version,
                            run_num,
                            dataset_name,
                            mapping_direction,
                        )
                    )
                except Exception:
                    continue

        # Sort by run number (descending)
        results.sort(key=lambda r: r.run_number or 0, reverse=True)
        return results

    def get_latest_run(
        self,
        engine_type: str,
        engine_version: int,
        dataset_name: str,
        mapping_direction: str = "inverse",
    ) -> DiagnosticReport:
        runs = self.get_all_runs(
            engine_type, engine_version, dataset_name, mapping_direction
        )
        if not runs:
            raise FileNotFoundError(
                f"No diagnostic runs found for {engine_type} v{engine_version}."
            )
        return runs[0]

    def get_batch(
        self,
        engines: list[Any],
        dataset_name: str,
        mapping_direction: str = "inverse",
    ) -> dict[str, DiagnosticReport]:
        """
        Fetches multiple runs. Expects objects with .type, .version, and .run_number.
        """
        results_map = {}
        for engine_req in engines:
            display_name = f"{engine_req.type} (v{engine_req.version})"
            if engine_req.run_number is None:
                results_map[display_name] = self.get_latest_run(
                    engine_req.type,
                    engine_req.version,
                    dataset_name,
                    mapping_direction,
                )
            else:
                results_map[display_name] = self.load(
                    engine_req.type,
                    engine_req.version,
                    engine_req.run_number,
                    dataset_name,
                    mapping_direction,
                )
        return results_map
