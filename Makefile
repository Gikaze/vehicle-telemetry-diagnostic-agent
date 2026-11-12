# Developer commands. Keep this list in sync with the "Commands" section of CLAUDE.md.

.DEFAULT_GOAL := help
SHELL := /bin/bash

COMPOSE_FILE := compose.yaml
# Pinned LocalStack image (NFR-07). Passed to `lstk start --image`, which applies it to one start
# without writing the lstk configuration. Update the tag deliberately, never to `latest`.
LOCALSTACK_IMAGE := localstack/localstack-pro:2026.9.1

.PHONY: help setup lint test test-int test-24h up down schema vehicle-day stream

# make test-24h: fleet size and cut of the fleet run (UNTIL=HH:MM stops day 0 there; empty: 24 h),
# and the file the run's JSON summary is written to (empty: a temporary file, removed after).
VEHICLES ?= 100
UNTIL ?=
SUMMARY ?=

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

stream: ## Run the telemetry stream; settings from SIM_* variables or ARGS="--target file --out x.jsonl"
	@uv run vtda-simulator $(ARGS)

test: ## Run unit tests with coverage
	uv run pytest tests/unit --cov

# pytest exits 0 when the test is skipped or deselected, so the target also requires the summary
# file, which the test writes only after every assertion has passed: no summary, no green.
test-24h: ## Fleet run (100 vans, 24 h; UNTIL=09:00 cuts it): every message validated and checked
	@summary="$(SUMMARY)"; temporary=; \
	if [ -z "$$summary" ]; then summary=$$(mktemp); temporary=1; fi; \
	rm -f "$$summary"; \
	TEST_24H_VEHICLES=$(VEHICLES) TEST_24H_UNTIL=$(UNTIL) TEST_24H_SUMMARY="$$summary" \
		uv run pytest tests/slow -m slow -s -p no:cacheprovider; status=$$?; \
	if [ $$status -eq 0 ] && [ ! -s "$$summary" ]; then \
		echo "make test-24h: the fleet run did not run (no summary written): failing" >&2; status=1; \
	fi; \
	if [ -n "$$temporary" ]; then rm -f "$$summary"; fi; \
	exit $$status

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
