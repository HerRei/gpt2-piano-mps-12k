PYTHON ?= python3

.PHONY: test smoke-test pipeline pipeline-dry-run pipeline-epoch2 pipeline-epoch4 profiles docs-example docs-serve

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

docs-example:
	/opt/homebrew/anaconda3/envs/gpt2piano/bin/python scripts/build_docs_example.py

docs-serve:
	$(PYTHON) -m http.server 8000 -d docs
