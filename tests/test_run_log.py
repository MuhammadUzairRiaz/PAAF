"""Every pipeline run leaves paaf_run.log and (when the optimiser ran) energies.txt."""
import logging
from pathlib import Path

import pytest

from paaf.run_log import (ENERGY_FILE_NAME, RUN_LOG_NAME, record_energy,
                          run_log, write_energy_table)


def test_run_log_captures_paaf_logger_and_energies(tmp_path):
    lg = logging.getLogger("paaf.something")
    with run_log(tmp_path) as info:
        lg.info("hello from inside the run")
        record_energy("PBS_20", "MMFF94", 234.7, 128.4, 483)
        record_energy("PBS", "MMFF94", 12.0, 10.5, 25)
    text = (tmp_path / RUN_LOG_NAME).read_text()
    assert "hello from inside the run" in text and "PAAF run started" in text
    assert info["log_file"].endswith(RUN_LOG_NAME)
    e = Path(info["energy_file"]).read_text()
    assert "PBS_20" in e and "MMFF94" in e and "-106.3000" in e
    # energies are not recorded outside a run
    record_energy("x", "UFF", 1.0, 0.0)
    assert "x " not in (tmp_path / ENERGY_FILE_NAME).read_text()
    # handler was detached
    assert not any(isinstance(h, logging.FileHandler) and RUN_LOG_NAME in getattr(h, "baseFilename", "")
                   for h in logging.getLogger("paaf").handlers)


def test_run_log_records_failure_and_still_detaches(tmp_path):
    with pytest.raises(RuntimeError):
        with run_log(tmp_path):
            raise RuntimeError("boom")
    assert "RuntimeError: boom" in (tmp_path / RUN_LOG_NAME).read_text()
    assert not (tmp_path / ENERGY_FILE_NAME).exists()


def test_no_energy_file_when_nothing_recorded(tmp_path):
    with run_log(tmp_path) as info:
        pass
    assert info["energy_file"] is None
    assert write_energy_table(tmp_path / "e.txt", []) is None


def test_pipeline_and_optimizer_are_wired():
    src = Path("paaf/pipeline.py").read_text()
    assert "with run_log(out_dir) as rl:" in src and 'result["log_file"]' in src
    assert "record_energy(" in Path("paaf/optimizer.py").read_text()
