SHELL := /bin/bash

.PHONY: test

test:
	pytest tests
