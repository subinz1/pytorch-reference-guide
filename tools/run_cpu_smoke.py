#!/usr/bin/env python3
"""Run the representative CPU examples listed in the smoke manifest."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("cpu_smoke_manifest.json")
MAX_OUTPUT_CHARS = 4_000


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def parse_manifest(path: Path) -> tuple[int, list[dict[str, object]]]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"manifest does not exist: {path}")
    except json.JSONDecodeError as error:
        fail(f"invalid manifest JSON: {error}")

    if not isinstance(manifest, dict):
        fail("manifest must be an object")
    default_timeout = manifest.get("default_timeout_seconds")
    examples = manifest.get("examples")
    if not isinstance(default_timeout, int) or default_timeout <= 0:
        fail("default_timeout_seconds must be a positive integer")
    if not isinstance(examples, list) or not examples:
        fail("examples must be a non-empty list")
    return default_timeout, examples


def validate_example(
    example: object, default_timeout: int
) -> tuple[str, Path, int]:
    if not isinstance(example, dict):
        fail("each example must be an object")

    relative_path = example.get("path")
    timeout = example.get("timeout_seconds", default_timeout)
    if not isinstance(relative_path, str) or not relative_path:
        fail("each example needs a non-empty path")
    if not isinstance(timeout, int) or timeout <= 0:
        fail(f"{relative_path}: timeout_seconds must be a positive integer")

    candidate = (REPO_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError:
        fail(f"{relative_path}: example must be inside the repository")
    if not candidate.is_file():
        fail(f"{relative_path}: example does not exist")
    return relative_path, candidate, timeout


def selected_examples(
    examples: list[dict[str, object]], only: set[str], default_timeout: int
) -> list[tuple[str, Path, int]]:
    validated = [validate_example(example, default_timeout) for example in examples]
    validated_paths = {relative_path for relative_path, _, _ in validated}
    if len(validated_paths) != len(validated):
        fail("the manifest contains duplicate paths")
    missing_paths = only.difference(validated_paths)
    if missing_paths:
        fail(f"not listed in the manifest: {', '.join(sorted(missing_paths))}")
    return [entry for entry in validated if not only or entry[0] in only]


def output_tail(output: str) -> str:
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    return f"... output truncated ...\n{output[-MAX_OUTPUT_CHARS:]}"


def run_example(relative_path: str, path: Path, timeout: int) -> bool:
    start = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        elapsed = time.monotonic() - start
        print(f"FAIL {relative_path} ({elapsed:.1f}s): timed out after {timeout}s")
        if error.stdout:
            print(output_tail(error.stdout))
        if error.stderr:
            print(output_tail(error.stderr), file=sys.stderr)
        return False

    elapsed = time.monotonic() - start
    if result.returncode == 0:
        print(f"PASS {relative_path} ({elapsed:.1f}s)")
        return True

    print(f"FAIL {relative_path} ({elapsed:.1f}s): exit code {result.returncode}")
    if result.stdout:
        print(output_tail(result.stdout))
    if result.stderr:
        print(output_tail(result.stderr), file=sys.stderr)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--only", action="append", default=[], metavar="PATH")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    default_timeout, examples = parse_manifest(args.manifest)
    selected = selected_examples(examples, set(args.only), default_timeout)
    if args.list:
        for relative_path, _, timeout in selected:
            print(f"{relative_path} (timeout: {timeout}s)")
        return

    failed = [
        relative_path
        for relative_path, path, timeout in selected
        if not run_example(relative_path, path, timeout)
    ]
    if failed:
        print(f"{len(failed)}/{len(selected)} CPU smoke examples failed", file=sys.stderr)
        raise SystemExit(1)
    print(f"{len(selected)} CPU smoke examples passed")


if __name__ == "__main__":
    main()
