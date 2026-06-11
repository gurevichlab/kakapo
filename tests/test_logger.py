"""Unit tests for the centralized logging system of kakapo.

Tests cover verbosity levels, file handler setup, log propagation, and preventing
duplicate handlers.
"""

import logging
from pathlib import Path

from kakapo.utils.logger import get_logger, setup_logging


def test_get_logger() -> None:
    """Test that get_logger returns the correct logger instance."""
    logger = get_logger("kakapo.test_module")
    assert logger.name == "kakapo.test_module"
    assert isinstance(logger, logging.Logger)


def test_setup_logging_defaults() -> None:
    """Test setup_logging with default parameters."""
    logger = setup_logging()
    assert logger.name == "kakapo"
    assert logger.level == logging.WARNING
    assert not logger.propagate

    # Should have exactly one StreamHandler (stdout)
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_setup_logging_verbosity_levels() -> None:
    """Test setup_logging verbosity mapping."""
    # Verbosity 0 -> WARNING
    logger = setup_logging(verbosity=0)
    assert logger.level == logging.WARNING

    # Verbosity 1 -> INFO
    logger = setup_logging(verbosity=1)
    assert logger.level == logging.INFO

    # Verbosity 2 -> DEBUG
    logger = setup_logging(verbosity=2)
    assert logger.level == logging.DEBUG

    # Verbosity out-of-bounds -> DEBUG (fallback)
    logger = setup_logging(verbosity=99)
    assert logger.level == logging.DEBUG


def test_setup_logging_with_file(tmp_path: Path) -> None:
    """Test setup_logging with a file path handler."""
    log_file = tmp_path / "logs" / "test.log"
    logger = setup_logging(verbosity=1, log_file=log_file)

    # Should have two handlers: stream and file
    assert len(logger.handlers) == 2
    handlers_types = {type(h) for h in logger.handlers}
    assert logging.StreamHandler in handlers_types
    assert logging.FileHandler in handlers_types

    # Log a message and check if file exists and has content
    test_message = "Test logging to file output"
    logger.info(test_message)

    # Close handlers to release the file lock before reading
    for handler in logger.handlers:
        handler.close()

    assert log_file.exists()
    log_content = log_file.read_text(encoding="utf-8")
    assert test_message in log_content


def test_setup_logging_prevents_duplicate_handlers() -> None:
    """Test that multiple setup_logging calls clear previous handlers."""
    # First setup
    logger = setup_logging(verbosity=1)
    assert len(logger.handlers) == 1

    # Second setup
    logger = setup_logging(verbosity=2)
    assert len(logger.handlers) == 1


def test_setup_logging_clears_file(tmp_path: Path) -> None:
    """Test that setup_logging clears/truncates the log file if it already exists."""
    log_file = tmp_path / "test_clear.log"
    log_file.write_text("pre-existing content", encoding="utf-8")

    logger = setup_logging(verbosity=1, log_file=log_file)
    logger.info("new content")

    # Close handlers to release file lock before reading
    for handler in logger.handlers:
        handler.close()

    log_content = log_file.read_text(encoding="utf-8")
    assert "pre-existing content" not in log_content
    assert "new content" in log_content

