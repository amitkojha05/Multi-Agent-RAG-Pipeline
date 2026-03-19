"""tools/code_executor.py

CodeExecutor runs Python snippets in a sandboxed subprocess.
"""
import os
import subprocess
import tempfile
from pathlib import Path


BLOCKED_IMPORTS = [
    "import os",
    "import sys",
    "import subprocess",
    "import socket",
    "import requests",
    "import urllib",
    "import http",
    "import ftplib",
    "import smtplib",
    "import paramiko",
    "import ctypes",
    "import cffi",
    "from os",
    "from sys",
    "from subprocess",
    "from socket",
    "__import__(",
]


class CodeExecutor:
    def __init__(self) -> None:
        self.timeout = int(os.getenv("CODE_EXEC_TIMEOUT", "5"))

    def execute(self, code: str) -> str:
        """Execute code snippet. Returns stdout or error string."""
        code_lower = code.lower()
        for pattern in BLOCKED_IMPORTS:
            if pattern in code_lower:
                return f"Error: blocked pattern detected: {pattern}"

        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "script.py"
            script.write_text(code)

            try:
                result = subprocess.run(
                    ["python", str(script)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env={
                        "PATH": os.getenv("PATH", ""),
                        "HOME": tmpdir,
                        "PYTHONPATH": "",
                    },
                    cwd=tmpdir,
                )
                if result.returncode == 0:
                    return result.stdout[:2000]
                return f"Error (exit {result.returncode}): {result.stderr[:500]}"
            except subprocess.TimeoutExpired:
                return f"Error: execution timed out after {self.timeout}s"
            except Exception as e:  # pragma: no cover - error path
                return f"Error: {e}"

