.PHONY: setup test lint demo benchmark clean

PYTHON := python
PIP := pip

setup:
	$(PIP) install -r requirements.txt
	@echo "Setup complete."

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=src/speechmirror --cov-report=term-missing

lint:
	ruff check src/ scripts/ tests/
	black --check src/ scripts/ tests/

format:
	ruff check --fix src/ scripts/ tests/
	black src/ scripts/ tests/

demo:
	cd app && $(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
	@echo "API running at http://localhost:8000"
	@echo "Dashboard: open app/dashboard in browser"

benchmark-val:
	$(PYTHON) scripts/benchmark.py --split val --seed 42

benchmark-test:
	$(PYTHON) scripts/benchmark.py --split test --seed 42

perturb:
	$(PYTHON) scripts/generate_perturbations.py --config configs/perturb.json --seed 7

validate:
	$(PYTHON) scripts/validate_dataset.py --manifest data/manifests/manifest.csv

stats:
	$(PYTHON) scripts/stats_report.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
