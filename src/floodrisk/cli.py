"""Interface de linha de comando.

    python -m floodrisk stages
    python -m floodrisk run info
    python -m floodrisk run bootstrap --config configs/experimento_x.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, find_repo_root, load_config
from .logging_setup import setup_logging
from .pipeline import STAGES, StageNotImplementedError, run_stage


def _load_env() -> None:
    """Carrega o .env da raiz do repositório, se existir.

    Feito aqui, num lugar só: nenhum módulo deve ir atrás de arquivo de segredo
    por conta própria. Variável já presente no ambiente tem precedência sobre o
    arquivo — útil em CI.
    """
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    try:
        env_file = find_repo_root() / ".env"
    except RuntimeError:
        return
    if env_file.exists():
        load_dotenv(env_file, override=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="floodrisk",
        description="Pipeline de suscetibilidade a alagamentos urbanos em Curitiba.",
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML de configuração")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nível de log (padrão: FLOODRISK_LOG_LEVEL ou INFO)",
    )
    parser.add_argument(
        "--log-file", action="store_true", help="Também grava o log em paths.logs"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stages", help="Lista os estágios e seus status")

    run = sub.add_parser("run", help="Executa um estágio")
    run.add_argument("stage", help="Nome do estágio (ver 'stages')")

    return parser


def _print_stages() -> None:
    current_phase = -1
    for stage in STAGES:
        if stage.phase != current_phase:
            current_phase = stage.phase
            print(f"\nFase {current_phase}")
        status = "ok      " if stage.implemented else "pendente"
        print(f"  [{status}] {stage.name:<22} {stage.help}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "stages":
        _print_stages()
        return 0

    _load_env()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"erro de configuração: {exc}", file=sys.stderr)
        return 2

    log_dir = config.path("logs") if args.log_file else None
    setup_logging(level=args.log_level, log_dir=log_dir, stage=args.stage)

    try:
        run_stage(args.stage, config)
    except StageNotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except KeyError as exc:
        print(f"{exc}. Rode 'floodrisk stages' para ver os disponíveis.", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
