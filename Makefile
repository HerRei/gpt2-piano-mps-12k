PYTHON ?= python3

.PHONY: test pipeline pipeline-dry-run pipeline-epoch2 pipeline-epoch4 profiles docs-serve

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

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

docs-serve:
	$(PYTHON) -m http.server 8000 -d docs
