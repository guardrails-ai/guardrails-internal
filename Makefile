MKDOCS_SERVE_ADDR ?= localhost:8000 # Default address for mkdocs serve, format: <host>:<port>, override with `make docs-serve MKDOCS_SERVE_ADDR=<host>:<port>`

autoformat:
	black guardrails/ tests/
	isort --atomic guardrails/ tests/
	docformatter --in-place --recursive guardrails tests

lint:
	isort -c guardrails/ tests/
	black guardrails/ tests/ --check
	flake8 guardrails/ tests/

test:
	pytest tests/

test-basic:
	set -e
	python -c "import guardrails as gd"
	python -c "import guardrails.version as mversion"

test-cov:
	pytest tests/ --cov=./guardrails/ --cov-report=xml

view-test-cov:
	pytest tests/ --cov=./guardrails/ --cov-report html && open htmlcov/index.html

docs-serve:
	mkdocs serve -a $(MKDOCS_SERVE_ADDR)

docs-deploy:
	mkdocs gh-deploy

dev:
	pip install -e ".[dev]"

full:
	pip install -e ".[all]"

all: autoformat lint docs test

pydocs:
	# sphinx-apidoc -o docs/pydocs/config guardrails;
	sphinx-build -b html docs/pydocs/ docs/pydocs/build;

jupyter-html:
	jupyter nbconvert --to html docs/**/*.ipynb

docs-gen:
	# sphinx-apidoc -o docs/pydocs/config guardrails;
	sphinx-build -b html docs/pydocs/ docs/pydocs/build;
	nbdoc_build