# Developer commands. Keep this list in sync with the "Commands" section of CLAUDE.md.

.DEFAULT_GOAL := help
SHELL := /bin/bash

COMPOSE_FILE := compose.yaml

.PHONY: help setup lint test test-int up down

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup: ## Install dependencies (uv sync) and the git hooks
	uv sync --locked
	pre-commit install

lint: ## Run all linters: ruff, terraform fmt, tflint, hadolint, gitleaks, file hygiene
	pre-commit run --all-files

test: ## Run unit tests with coverage
	uv run pytest tests/unit --cov

test-int: ## Run integration tests against LocalStack (requires lstk running)
	@lstk status --non-interactive >/dev/null || { echo "LocalStack is not running: run 'make up' first"; exit 1; }
	@uv run pytest tests/integration -m integration; status=$$?; \
	if [ $$status -eq 5 ]; then echo "No integration tests collected yet."; exit 0; fi; \
	exit $$status

up: ## Start LocalStack and local services
	lstk start --non-interactive --type aws
	@if [ -f $(COMPOSE_FILE) ]; then docker compose -f $(COMPOSE_FILE) up -d; fi

down: ## Stop local services and LocalStack
	@if [ -f $(COMPOSE_FILE) ]; then docker compose -f $(COMPOSE_FILE) down; fi
	lstk stop --non-interactive
