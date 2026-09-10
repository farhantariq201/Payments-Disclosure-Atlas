.PHONY: install test lint fetch parse goldset classify benchmark analyze memo all

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

fetch:
	pda fetch

parse:
	pda parse

goldset:
	pda goldset --n 400

classify:
	pda classify

benchmark:
	pda benchmark

analyze:
	pda analyze

memo:
	pda memo

# Everything that does not require hand-labelling.
all: fetch parse classify analyze memo
