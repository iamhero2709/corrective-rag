.PHONY: install install-cli test test-fast serve docker lint query demo index benchmark download-models

install:
	pip install -e ".[dev]" && python -m spacy download en_core_web_sm

install-cli:
	pip install -e ".[dev,cli]" && python -m spacy download en_core_web_sm

install-all:
	pip install -e ".[dev,cli,hybrid]" && python -m spacy download en_core_web_sm

test:
	pytest -q

test-fast:
	pytest -q tests/test_hrr.py tests/test_graph.py tests/test_config.py tests/test_evaluate.py

serve:
	uvicorn server:app --host 0.0.0.0 --port 8000 --reload

docker:
	docker compose up --build

lint:
	ruff check src/ server.py tests/ cli.py

query:
	rag query "$(Q)" --model $(M)

demo:
	rag demo --model $(M)

index:
	rag index $(PATH)

benchmark:
	rag benchmark --dataset $(D) --samples $(N)

download-models:
	python scripts/download_models.py --models $(M)
