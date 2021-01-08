SHELL := /bin/bash

.PHONY: image install install-dev clean test-local test-docker pre-commit

test:
    pytest tests
