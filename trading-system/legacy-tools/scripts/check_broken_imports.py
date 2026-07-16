#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ast
import os
import sys
from collections import defaultdict
from pathlib import Path

STDLIB = {
    "os", "sys", "re", "json", "ast", "time", "math", "hashlib",
    "logging", "threading", "collections", "datetime", "pathlib",
    "typing", "dataclasses", "functools", "itertools", "copy",
    "subprocess", "shutil", "tempfile", "traceback", "inspect",
    "importlib", "textwrap", "string", "random", "uuid", "enum",
    "abc", "io", "struct", "sqlite3", "socket", "http", "urllib",
    "unittest", "argparse", "asyncio", "queue", "concurrent",
    "contextlib", "warnings", "signal", "select", "ssl", "secrets",
    "hmac", "base64", "csv", "pickle", "platform", "ctypes",
}

EXTERNAL = {
    "numpy", "pandas", "torch", "tensorflow", "sklearn", "scipy",
    "matplotlib", "requests", "flask", "fastapi", "django", "pydantic",
    "openai", "anthropic", "tiktoken", "chromadb", "langchain",
    "PIL", "cv2", "psutil", "rich", "click", "pyyaml", "yaml",
    "dotenv", "tqdm", "aiohttp", "httpx", "sqlalchemy",
}


def find_issues(project_root: str = "."):
    root = Path(project_root)
    project_modules = set()
    for f in root.rglob("*.py"):
        if "__pycache__" in f.parts:
            continue
        rel = str(f.relative_to(root))
        mod_name = rel.replace(os.sep, ".").replace("/", ".").replace(".py", "")
        project_modules.add(mod_name)

    issues = []
    for f in root.rglob("*.py"):
        if "__pycache__" in f.parts:
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        mod = str(f.relative_to(root))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in STDLIB and top not in EXTERNAL:
                        if not any(pm.startswith(top) for pm in project_modules):
                            if not (root / f"{top}.py").exists() and not (root / top / "__init__.py").exists():
                                issues.append(f"模块不存在: {alias.name} ({mod}:{node.lineno})")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    if top not in STDLIB and top not in EXTERNAL:
                        if not any(pm.startswith(top) for pm in project_modules):
                            if not (root / f"{top}.py").exists() and not (root / top / "__init__.py").exists():
                                issues.append(f"模块不存在: {node.module} ({mod}:{node.lineno})")
    return issues


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    issues = find_issues(root)
    if issues:
        for issue in issues[:50]:
            print(issue)
        print(f"\n共发现 {len(issues)} 个依赖断裂问题")
        sys.exit(1)
    print("无依赖断裂问题")


if __name__ == "__main__":
    main()
