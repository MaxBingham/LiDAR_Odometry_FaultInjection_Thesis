.PHONY: install clean rebuild sync test hooks help

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Full installation (dependencies + KISS-ICP)
	pip install -e .
	pip install scikit-build-core pybind11
	export PATH="$$(pwd)/venv/bin:$$PATH" && \
	pip install --no-build-isolation -e external/kiss-icp/python

clean:  ## Remove build artifacts and cache
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf external/kiss-icp/python/build
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.so' -delete

rebuild: clean install  ## Clean rebuild of everything

sync:  ## Quick sync after branch switch (use this most often)
	@echo "Syncing environment with current branch..."
	@pip install -e . --no-deps --quiet
	@echo "✅ Environment synced"

sync-full:  ## Full sync including dependencies
	@echo "Full sync with dependency resolution..."
	@pip install -e .
	@echo "✅ Full sync complete"

rebuild-kiss-icp:  ## Rebuild KISS-ICP only (after modifications)
	rm -rf external/kiss-icp/python/build
	export PATH="$$(pwd)/venv/bin:$$PATH" && \
	pip install --no-build-isolation -e external/kiss-icp/python

hooks:  ## Install Git hooks (post-checkout auto-sync)
	./scripts/install-hooks.sh

test:  ## Run tests (if you add them)
	pytest tests/ -v

run-fog:  ## Run fog model on sequence 03 (example)
	python3 -m lfl.runner --fault_model fog --sequence 03 --fog_metric distance

run-rain:  ## Run rain model on sequence 03 (example)
	python3 -m lfl.runner --fault_model rain --sequence 03

.DEFAULT_GOAL := help
