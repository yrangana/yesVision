install:
		pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
		python -m pytest -vv

lint:
		pylint --disable=R,C

all: install lint test