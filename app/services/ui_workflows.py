from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def run_project_command(arguments: list[str], timeout: int = 1800) -> dict[str, Any]:
    """Run an explicit user-requested project workflow with the active Python interpreter."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        completed = subprocess.run(
            [sys.executable, *arguments],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": int(completed.returncode),
            "stdout": completed.stdout or "",
            "stderr": completed.stderr or "",
            "command": [sys.executable, *arguments],
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", errors="replace")
        return {
            "returncode": -1,
            "stdout": str(out),
            "stderr": str(err) + f"\n任务超过 {timeout} 秒，已停止等待。",
            "command": [sys.executable, *arguments],
            "timeout": True,
        }
    except OSError as exc:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"启动任务失败：{type(exc).__name__}: {exc}",
            "command": [sys.executable, *arguments],
            "timeout": False,
        }
