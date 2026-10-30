# Developer commands. Keep this list in sync with the "Commands" section of CLAUDE.md.

.DEFAULT_GOAL := help
SHELL := /bin/bash

COMPOSE_FILE := compose.yaml
# Pinned LocalStack image (NFR-07). Passed to `lstk start --image`, which applies it to one start
# without writing the lstk configuration. Update the tag deliberately, never to `latest`.
LOCALSTACK_IMAGE := localstack/localstack-pro:2026.9.1

.PHONY: help setup lint test test-int up down schema vehicle-day

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install dependencies (uv sync) and the git hooks
	uv sync --locked
	pre-commit install

lint: ## Run all linters: ruff, terraform fmt, tflint, hadolint, gitleaks, file hygiene
	pre-commit run --all-files

schema: ## Regenerate the telemetry JSON Schema from simulator/catalogue.yaml
	uv run python -m simulator.schema

vehicle-day: ## Run and check one fleet day (10 vans, seed 42); plot VAN-001's day to docs/assets/
	uv run python simulator/tools/vehicle_day.py --seed 42 --vehicles 10 --vehicle VAN-001 \
		--out docs/assets/m1-vehicle-day.png

test: ## Run unit tests with coverage
	uv run pytest tests/unit --cov

test-int: ## Run integration tests against LocalStack (requires lstk running)
	@lstk status --non-interactive >/dev/null || { echo "LocalStack is not running: run 'make up' first"; exit 1; }
	@uv run pytest tests/integration -m integration; status=$$?; \
	if [ $$status -eq 5 ]; then echo "No integration tests collected yet."; exit 0; fi; \
	exit $$status

up: ## Start LocalStack and local services
	lstk start --non-interactive --type aws --image $(LOCALSTACK_IMAGE)
	@if [ -f $(COMPOSE_FILE) ]; then docker compose -f $(COMPOSE_FILE) up -d; fi

down: ## Stop local services and LocalStack
	@if [ -f $(COMPOSE_FILE) ]; then docker compose -f $(COMPOSE_FILE) down; fi
	lstk stop --non-interactive
