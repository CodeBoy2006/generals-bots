.PHONY: install remote test build clean

# Install package
install:
	uv sync --extra dev

remote:
	uv run python -m examples.client_example

###################
# Developer tools #
###################

test:
	uv run pytest

build:
	uv build

clean:
	rm -rf build dist *.egg-info
