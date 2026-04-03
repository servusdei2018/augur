CARGO ?= cargo
.DEFAULT_GOAL := help

.PHONY: help lint format test build

help: ## Display this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

format: ## Apply rustfmt to all crates
	$(CARGO) fmt --all

test: ## Run the test suite
	$(CARGO) test

build: ## Build the project (debug)
	$(CARGO) build
