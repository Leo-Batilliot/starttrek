VENV   := .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
SEED   ?= 0

help:
	@echo "Targets:"
	@echo "  make install                              — create .venv and install deps"
	@echo "  make train CONFIG=<path> [SEED=<int>]     — run training (default SEED=0)"
	@echo "  make eval RUN=<path>                      — evaluate a trained run"
	@echo "  make test                                 — run the unittest suite"
	@echo "  make reproduce                            — execute reproduce.sh"
	@echo "  make clean                                — remove generated artifacts"

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

train:
	@if [ -z "$(CONFIG)" ]; then echo "ERROR: CONFIG=<path> is required"; exit 1; fi
	$(PY) train.py --config $(CONFIG) --seed $(SEED)

eval:
	@if [ -z "$(RUN)" ]; then echo "ERROR: RUN=<path> is required"; exit 1; fi
	$(PY) eval.py --run $(RUN)

test:
	$(PY) -m unittest discover tests/

reproduce:
	bash reproduce.sh

clean:
	rm -rf runs/ __pycache__/ */__pycache__/ **/__pycache__/
	rm -rf .pytest_cache/ .ruff_cache/ *.egg-info/ build/ dist/
	@echo "Note: .venv/ is NOT deleted by default. Run 'make clean-venv' to remove it."

clean-venv:
	rm -rf $(VENV)

.PHONY: install train eval test reproduce clean help