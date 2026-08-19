"""Subprocess entry point for one persistent combat environment."""
from __future__ import annotations

import os
import traceback
from typing import Any

from .env import MultiUAVCombatEnv


def combat_environment_worker(connection: Any, config: Any) -> None:
    """Own one environment and serve reset/step commands over a pipe."""
    try:
        environment = MultiUAVCombatEnv(config)
        connection.send(("ready", os.getpid()))
        while True:
            command, payload = connection.recv()
            if command == "reset":
                observation, _ = environment.reset(int(payload))
                connection.send(("ok", (observation, environment.red_alive_mask)))
            elif command == "step":
                connection.send(("ok", environment.step(payload)))
            elif command == "close":
                connection.send(("ok", None))
                break
            else:
                raise RuntimeError(f"unknown environment-worker command: {command}")
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


__all__ = ["combat_environment_worker"]
