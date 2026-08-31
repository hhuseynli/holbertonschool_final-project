.PHONY: install demo full test app dataset maup harvest clean

install:
	pip install -r requirements.txt
	pip install -e .

# Quick end-to-end demo: smaller dataset, fewer epochs (~2-4 min)
demo:
	python scripts/run_pipeline.py --fast

# Full pipeline with the STGCN forecaster
full:
	python scripts/run_pipeline.py

dataset:
	python scripts/make_dataset.py

# Compare units of analysis (H3 vs adaptive KD-tree vs market regions)
# under the project's leakage-proof validation protocols.
maup:
	python scripts/maup_study.py

# Fetch real listings from bina.az (polite: 1 req/s, cached, robots-obeying)
harvest:
	python scripts/harvest_real.py --limit 500

test:
	python -m pytest tests/ -q

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf artifacts data/processed .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
