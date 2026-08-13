"""CLI delivery-adapter tests using typer's CliRunner."""

import re

import pytest
import typer
from typer.testing import CliRunner

from datalus.cli import (
    _resolve_checkpoint_path,
    _validated_verbose,
    app,
)

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


runner = CliRunner()

EXPECTED_COMMANDS = {
    "ingest",
    "train",
    "sample",
    "augment",
    "balance",
    "inpaint",
    "counterfactual",
    "audit",
    "export-onnx",
    "serve",
    "streamlit",
}


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in EXPECTED_COMMANDS:
        assert command in result.output


def test_streamlit_command_is_registered_as_streamlit():
    assert "streamlit" in EXPECTED_COMMANDS
    result = runner.invoke(app, ["streamlit", "--help"])
    assert result.exit_code == 0
    assert "Streamlit interface" in result.output


def test_streamlit_app_alias_is_not_registered():
    result = runner.invoke(app, ["streamlit-app", "--help"])
    assert result.exit_code == 2


def test_invalid_global_verbose_is_rejected():
    result = runner.invoke(app, ["--verbose", "BOGUS", "train", "--help"])
    assert result.exit_code == 2
    assert "not one of WARNING, INFO, DEBUG" in result.output


def test_invalid_command_verbose_is_rejected():
    result = runner.invoke(app, ["train", "--verbose", "LOUD", "a", "b", "c"])
    assert result.exit_code == 2
    assert "not one of WARNING, INFO, DEBUG" in result.output


def test_valid_verbose_before_subcommand_accepted():
    result = runner.invoke(app, ["--verbose", "INFO", "train", "--help"])
    assert result.exit_code == 0


def test_valid_verbose_after_subcommand_accepted():
    result = runner.invoke(app, ["train", "--verbose", "INFO", "--help"])
    assert result.exit_code == 0


def test_invalid_mia_mode_is_rejected():
    result = runner.invoke(app, ["audit", "--mia-mode", "strict", "a", "b", "c", "d"])
    assert result.exit_code == 2
    assert "not one of 'release', 'ci_lite'" in result.output


def test_valid_mia_mode_release_proceeds_to_io(tmp_path):
    for name in ("real.parquet", "syn.parquet"):
        (tmp_path / name).write_bytes(b"not-a-parquet")
    (tmp_path / "schema.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "audit",
            "--mia-mode",
            "release",
            str(tmp_path / "real.parquet"),
            str(tmp_path / "syn.parquet"),
            str(tmp_path / "schema.json"),
            str(tmp_path / "report.json"),
        ],
    )
    assert result.exit_code != 2
    assert not isinstance(result.exception, SystemExit)


def test_invalid_save_strategy_is_rejected():
    result = runner.invoke(app, ["train", "--save-strategy", "fancy", "a", "b", "c"])
    assert result.exit_code == 2
    assert "not one of 'all', 'latest', 'best'" in result.output


def test_invalid_checkpoint_source_is_rejected():
    result = runner.invoke(
        app,
        ["sample", "--checkpoint-source", "weird", "a", "b", "c"],
    )
    assert result.exit_code == 2
    assert "not one of 'latest', 'best'" in result.output


def test_train_help_exposes_checkpoint_cadence_flags():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)
    assert "--checkpoint-every-steps" in output
    assert "--save-every" in output
    assert "every N epochs" in output


def test_resolve_checkpoint_path_with_directory(tmp_path):
    (tmp_path / "checkpoint_best.pt").touch()
    resolved = _resolve_checkpoint_path(tmp_path, "best")
    assert resolved == tmp_path / "checkpoint_best.pt"


def test_resolve_checkpoint_path_missing_source_is_clean_error(tmp_path):
    (tmp_path / "checkpoint_best.pt").touch()
    with pytest.raises(typer.BadParameter, match="Checkpoint source not found"):
        _resolve_checkpoint_path(tmp_path, "latest")


def test_resolve_checkpoint_path_with_file_ignores_source(tmp_path):
    checkpoint = tmp_path / "custom.pt"
    checkpoint.touch()
    assert _resolve_checkpoint_path(checkpoint, "best") == checkpoint


def test_validate_verbose_accepts_known_levels():
    for level in ("WARNING", "INFO", "DEBUG"):
        assert _validated_verbose(level) == level


def test_validate_verbose_rejects_unknown_level():
    with pytest.raises(typer.BadParameter, match="not one of WARNING, INFO, DEBUG"):
        _validated_verbose("NOISY")
