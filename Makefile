.PHONY: sync test unit notebooks lint

sync:
	uv sync

unit:
	uv run pytest tests -v

notebooks:
	uv run pytest --nbmake --nbmake-timeout=600 notebooks exercises -v

test: unit notebooks

lint:
	uv run ruff check .
