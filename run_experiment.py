#!/usr/bin/env python3
"""Run the standalone PropKAG main pipeline with explicit experiment settings."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

from src.utils.env import read_env_file


ROOT = Path(__file__).resolve().parent
CLI_KEYS = {"initial_top_k", "top_k", "min_score", "graph_threshold", "feedback_candidates_per_query"}
POSITIVE_INTEGERS = {
    "LLM_CONTRIBUTION_WORKERS", "LLM_SYNTHESIS_WORKERS",
    "LLM_TYPE_INTEGRATION_WORKERS", "LLM_TYPE_INTEGRATION_TIMEOUT_SECONDS",
    "LLM_GROUP_NAMING_BATCH_SIZE", "LLM_GROUP_NAMING_WORKERS",
    "LLM_GAP_DIAGNOSIS_WORKERS", "LLM_CANDIDATE_FILTER_WORKERS",
    "LITERATURE_RETRIEVAL_WORKERS", "LITERATURE_RETRIEVAL_BATCH_SIZE",
    "MAX_GAP_DIRECTIONS_PER_GROUP",
    "EMBEDDING_DIMENSIONS", "EMBEDDING_BATCH_SIZE", "EMBEDDING_MAX_CHARS",
}
NONNEGATIVE_FLOATS = {
    "LITERATURE_RETRIEVAL_BATCH_DELAY_SECONDS",
    "LITERATURE_RETRIEVAL_RETRY_BASE_SECONDS",
    "LITERATURE_RETRIEVAL_RETRY_MAX_SECONDS",
}
ENV_KEYS = POSITIVE_INTEGERS | NONNEGATIVE_FLOATS | {
    "LLM_JSON_MODE", "GROUP_LEIDEN_RESOLUTION", "GROUP_LEIDEN_SEED", "GROUP_GRAPH_EDGE_RULE",
}
REQUIRED_SERVICES = (
    "LLM_API_KEY", "LLM_API_BASE_URL", "LLM_MODEL",
    "LLM_BASIC_API_KEY", "LLM_BASIC_API_BASE_URL", "LLM_BASIC_MODEL",
    "LITERATURE_SEARCH_API_KEY", "LITERATURE_SEARCH_API_ENDPOINT",
    "EMBEDDING_API_KEY", "EMBEDDING_API_BASE_URL", "EMBEDDING_MODEL",
)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"schema_version", "command_line", "environment"}:
        raise ValueError("Config must contain schema_version, command_line and environment only.")
    if type(config["schema_version"]) is not int or config["schema_version"] != 2:
        raise ValueError("Unsupported experiment configuration schema.")
    cli = config["command_line"]
    if not isinstance(cli, dict) or set(cli) != CLI_KEYS:
        raise ValueError("command_line must define initial_top_k, top_k, min_score, graph_threshold and feedback_candidates_per_query.")
    for key in ("initial_top_k", "top_k", "feedback_candidates_per_query"):
        if type(cli[key]) is not int or cli[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if cli["top_k"] > cli["initial_top_k"]:
        raise ValueError("top_k cannot exceed initial_top_k.")
    for key in ("min_score", "graph_threshold"):
        value = cli[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be a finite number between 0 and 1.")
    environment = config["environment"]
    if not isinstance(environment, dict):
        raise ValueError("environment must be an object.")
    if set(environment) != ENV_KEYS:
        raise ValueError("environment must contain the supported experiment settings only.")
    for key, value in environment.items():
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string.")
        if key in POSITIVE_INTEGERS and (not value.isdigit() or int(value) < 1):
            raise ValueError(f"{key} must be a positive integer.")
        if key == "MAX_GAP_DIRECTIONS_PER_GROUP" and int(value) > 3:
            raise ValueError("The manuscript permits at most three information gaps per draft.")
        if key in NONNEGATIVE_FLOATS | {"GROUP_LEIDEN_RESOLUTION"}:
            number = float(value)
            minimum = 0.01 if key == "GROUP_LEIDEN_RESOLUTION" else 0.0
            if not math.isfinite(number) or number < minimum:
                raise ValueError(f"{key} is outside its supported range.")
        if key == "GROUP_LEIDEN_SEED" and (not value.isdigit() or int(value) < 1):
            raise ValueError("GROUP_LEIDEN_SEED must be a positive integer.")
        if key == "LLM_JSON_MODE" and value != "true":
            raise ValueError("The main reproduction configuration requires LLM_JSON_MODE=true.")
        if key == "GROUP_GRAPH_EDGE_RULE" and value != "threshold":
            raise ValueError("The supported graph rule is threshold-only, as described in the manuscript.")
    if int(environment["EMBEDDING_MAX_CHARS"]) < 100:
        raise ValueError("EMBEDDING_MAX_CHARS must be at least 100.")
    return config


def validate_topic_id(value: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError("Topic IDs must be a single nonempty directory name without path separators.")
    return value


def read_topics(path: Path) -> list[tuple[str, str]]:
    topics: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" not in line:
            raise ValueError(f"Topic file line {line_number} must contain topic_id TAB query.")
        topic_id, query = line.split("\t", 1)
        topic_id = validate_topic_id(topic_id)
        query = query.strip()
        if not query:
            raise ValueError(f"Topic file line {line_number} has an empty query.")
        if topic_id.casefold() in seen:
            raise ValueError(f"Duplicate topic ID on line {line_number}.")
        seen.add(topic_id.casefold())
        topics.append((topic_id, query))
    if not topics:
        raise ValueError("Topic file contains no topic rows.")
    return topics


def select_topics(topics: list[tuple[str, str]], start_from: str) -> list[tuple[str, str]]:
    if start_from.isdigit():
        position = int(start_from)
        if 1 <= position <= len(topics):
            return topics[position - 1:]
        raise ValueError("--start-from is outside the topic file.")
    for index, (topic_id, _) in enumerate(topics):
        if topic_id.casefold() == start_from.casefold():
            return topics[index:]
    raise ValueError("--start-from does not match a topic ID.")


def command_for(topic: tuple[str, str], config: dict) -> list[str]:
    topic_id, query = topic
    command = [sys.executable, "-B", "-u", "-m", "src.main", "--query", query, "--topic-id", topic_id]
    for key, value in config["command_line"].items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    return command


def runtime_environment(config: dict) -> dict[str, str]:
    environment = dict(read_env_file(ROOT / ".env"))
    environment.update(os.environ)
    # Explicit experiment parameters override local and inherited numeric settings.
    environment.update(config["environment"])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def check_services(environment: dict[str, str]) -> None:
    missing = [key for key in REQUIRED_SERVICES if not environment.get(key, "").strip()]
    placeholders = [key for key in REQUIRED_SERVICES if any(
        marker in environment.get(key, "").casefold()
        for marker in ("your-", "replace_me", "replace-with", ".example")
    )]
    if missing or placeholders:
        # Report variable names only: never include credentials or service URLs.
        raise ValueError("Configure nonempty service settings in .env: " + ", ".join(sorted(set(missing + placeholders))))
    if importlib.util.find_spec("graspologic") is None:
        raise ValueError("Install dependencies first: python -m pip install -r requirements.txt")


def check_prompts() -> None:
    from src.orchestration.coordinator import Coordinator
    # The existing loader only reads project_root; avoid creating output folders.
    prompts = Coordinator._load_prompts(SimpleNamespace(project_root=ROOT))
    if any(not value.strip() for value in prompts.values()):
        raise ValueError("A required prompt is empty.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "experiment.json")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--query", help="User-supplied research topic or query.")
    source.add_argument("--topics-file", type=Path, help="User-supplied UTF-8 topic_id TAB query file.")
    parser.add_argument("--topic-id", default="T01")
    parser.add_argument("--start-from", default="1", help="1-based position or topic ID in --topics-file.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print commands without credentials, API calls or outputs.")
    parser.add_argument("--check", action="store_true", help="Check configuration, prompts, credentials and dependency presence without API calls.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        check_prompts()
        topics: list[tuple[str, str]] = []
        if args.topics_file is not None:
            topics = select_topics(read_topics(args.topics_file), args.start_from.strip())
        elif args.query is not None:
            if not args.query.strip():
                raise ValueError("--query cannot be empty.")
            if args.start_from != "1":
                raise ValueError("--start-from requires --topics-file.")
            topics = [(validate_topic_id(args.topic_id), args.query.strip())]
        elif not args.check:
            raise ValueError("Provide --query or --topics-file, or use --check.")
        environment = runtime_environment(config)
        if args.check or not args.dry_run:
            check_services(environment)
        if args.check:
            print("Local checks passed. No API request was made; service access and model availability are untested.")
            return 0
        print("Experiment configuration: " + str(args.config.resolve()), flush=True)
        print(f"Topics: {len(topics)}; initial_top_k={config['command_line']['initial_top_k']}; top_k={config['command_line']['top_k']}", flush=True)
        print(f"Feedback candidates/query: {config['command_line']['feedback_candidates_per_query']}; graph rule: {config['environment']['GROUP_GRAPH_EDGE_RULE']}", flush=True)
        failed = False
        for index, topic in enumerate(topics, 1):
            print(f"[{index}/{len(topics)}] {topic[0]}", flush=True)
            command = command_for(topic, config)
            if args.dry_run:
                print(shlex.join(command))
                continue
            completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
            if completed.returncode:
                print(f"Topic {topic[0]} failed (exit code {completed.returncode}).", file=sys.stderr)
                if not args.continue_on_error:
                    return completed.returncode if completed.returncode > 0 else 1
                failed = True
        return 1 if failed else 0
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
