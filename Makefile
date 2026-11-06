.PHONY: install train monitor monitor-drift ui test lint
install:
	pip install -r requirements.txt
train:
	python -m src.train
monitor:
	python -m src.monitor_run
monitor-drift:
	python -m src.monitor_run --drift
ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
test:
	pytest -q
lint:
	ruff check .
