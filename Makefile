PYTHON ?= python3

.PHONY: test smoke-test pipeline pipeline-dry-run pipeline-epoch2 pipeline-epoch4 profiles

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

smoke-test:
	$(PYTHON) -m unittest tests.test_dependency_smoke

pipeline:
	$(PYTHON) scripts/generation_pipeline.py

pipeline-dry-run:
	$(PYTHON) scripts/generation_pipeline.py --dry-run

pipeline-epoch2:
	$(PYTHON) scripts/generation_pipeline.py --epochs 2

pipeline-epoch4:
	$(PYTHON) scripts/generation_pipeline.py --epochs 4

profiles:
	$(PYTHON) scripts/generation_pipeline.py --list-profiles
