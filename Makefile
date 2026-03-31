.PHONY: setup setup-gke test lint lock

setup:
	uv venv --seed --python 3.12
	uv pip install --extra-index-url https://download.pytorch.org/whl/cpu -e .

setup-gke:
	@if [ -z "$(PROJECT)" ]; then echo "Usage: make setup-gke PROJECT=my-project [ZONE=us-west4-a] [CLUSTER=tpu]"; exit 1; fi
	@make setup
	bash gke/setup/setup_xpk.sh PROJECT=$(PROJECT) ZONE=$(or $(ZONE),us-west4-a) CLUSTER=$(or $(CLUSTER),tpu)
	bash gke/setup/setup_kaniko.sh PROJECT=$(PROJECT) ZONE=$(or $(ZONE),us-west4-a)
	@echo ""
	@echo "==> GKE setup complete. Submit jobs with:"
	@echo "    python gke/submit.py --infra gke/infra/v5e.yaml gke/jobs/train/my_job.sh"

test:
	uv run pytest tests/unit -x -v

lock:
	uv pip compile pyproject.toml -o requirements.txt

lint:
	uv run python -m py_compile src/megatext/trainers/pretrain.py
	uv run python -m py_compile src/megatext/configs/pyconfig.py
	uv run python -m py_compile src/megatext/conversion/convert.py
