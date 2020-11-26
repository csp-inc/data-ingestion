SHELL := /bin/bash

.PHONY: image install install-dev clean test-local test-docker pre-commit

docker-build:
	docker build -t bgs:experiment .

venv:
	virtualenv --python=python3.8 venv

install: venv
	source venv/bin/activate; \
	pip install -r requirements.txt

docker-jupyter:
	docker run \
	  -v `pwd`:`pwd` \
	  -w `pwd` \
	  -p 8888:8888 \
	  -it bgs:experiment  \
	  /bin/bash -c "jupyter lab --allow-root --no-browser --ip=0.0.0.0"

docker-run:
	docker run \
	  -v `pwd`:`pwd` \
	  -w `pwd` \
	  -p 8888:8888 \
	  -m 8g \
	  -it bgs:experiment  \
	  /bin/bash -c "/bin/bash"

pre-commit:
	source venv/bin/activate; \
	pre-commit run --all-files
