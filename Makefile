.PHONY: install dev test demo demo-approve demo-change demo-reject lint run clean

install:
	uv sync

dev:
	uv sync --extra dev

test:
	uv run pytest

demo: demo-approve

demo-approve:
	uv run python scripts/demo.py --scenario approve

demo-change:
	uv run python scripts/demo.py --scenario change

demo-reject:
	uv run python scripts/demo.py --scenario reject

lint:
	uv run ruff check src tests

run:
	uv run invoice-agent

clean:
	rm -rf out/*.pdf data/*.sqlite .pytest_cache .ruff_cache **/__pycache__
