"""QTS-Architect Application Entrypoint.

This module bootstraps the application using Hydra for configuration management.
All runtime parameters are externalized to YAML files in conf/.

Usage:
    # Default config (backtest mode)
    python -m qts_core.main

    # Override environment
    python -m qts_core.main env=prod

    # Multirun for hyperparameter sweeps
    python -m qts_core.main -m model=model_a,model_b
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import hydra
import structlog
from omegaconf import OmegaConf


if TYPE_CHECKING:
    from omegaconf import DictConfig

# ==============================================================================
# Constants
# ==============================================================================
APP_NAME: str = "QTS-Architect"
APP_VERSION: str = "0.1.0"


# ==============================================================================
# Logging Configuration
# ==============================================================================
def configure_logging(*, json_output: bool = True, log_level: str = "INFO") -> None:
    """Configure structlog for production-grade logging.

    Design Decisions:
    - JSON output for log aggregation (ELK, Datadog, etc.)
    - Context variables for request tracing
    - ISO timestamps for timezone-aware logs

    Args:
        json_output: If True, output JSON. If False, output human-readable logs.
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    # Set stdlib logging level (structlog wraps stdlib)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Processor chain: each processor transforms the event dict
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    if json_output:
        # Production: JSON for machine parsing
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        # Development: colored console output
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ==============================================================================
# Application Bootstrap
# ==============================================================================
@hydra.main(version_base=None, config_path="../../../conf", config_name="main")
def main(cfg: DictConfig) -> None:
    """Application entrypoint with Hydra configuration injection.

    Hydra provides:
    - Hierarchical config composition (defaults, overrides)
    - CLI overrides (e.g., python main.py model.learning_rate=0.01)
    - Automatic working directory management
    - Multirun for hyperparameter sweeps

    Args:
        cfg: Resolved configuration from Hydra.
    """
    # Determine logging mode from config
    is_debug: bool = bool(cfg.get("debug", False))
    env: str = str(cfg.get("env", "dev"))
    log_level: str = "DEBUG" if is_debug else "INFO"
    json_output: bool = env != "dev"  # Human-readable in dev, JSON in prod/staging

    configure_logging(json_output=json_output, log_level=log_level)
    log = structlog.get_logger()

    # Log startup with structured context
    log.info(
        "Initializing application",
        app=APP_NAME,
        version=APP_VERSION,
        env=env,
        debug=is_debug,
    )

    # Validate config resolution (catch missing interpolations)
    try:
        OmegaConf.resolve(cfg)
    except Exception as e:
        log.error("Configuration resolution failed", error=str(e))
        raise

    # Debug: print resolved config
    if is_debug:
        log.debug("Resolved configuration", config=OmegaConf.to_yaml(cfg))

    # -------------------------------------------------------------------------
    # Dependency Injection via Hydra instantiate()
    # -------------------------------------------------------------------------
    # Components are instantiated from _target_ in YAML configs.
    # Example:
    #   dataloader = hydra.utils.instantiate(cfg.dataset)
    #   model = hydra.utils.instantiate(cfg.model)
    #   supervisor = Supervisor(agents=[...], risk_agent=...)
    #
    # This placeholder demonstrates the pattern. Actual implementation
    # depends on the specific run mode (backtest, live, paper).
    # -------------------------------------------------------------------------

    log.info(
        "System ready",
        app=APP_NAME,
        version=APP_VERSION,
        status="INITIALIZED",
    )


if __name__ == "__main__":
    main()
