install:
	pip install -r requirements.txt

train:
	python scripts/train_model.py

run:
	streamlit run dashboards/streamlit_dashboard.py

test:
	pytest tests/

lint:
	ruff check .

format:
	ruff format .
