PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: setup data analysis site test deploy clean

setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -r requirements.txt

data:
	$(PY) scripts/get_data.py

# nuisance models (~11 min) -> readout (~5 min incl. CUPAC scores) -> power (~5 s) -> uplift (~20 min)
analysis:
	$(PY) scripts/nuisance.py
	$(PY) scripts/readout.py
	$(PY) scripts/power.py
	$(PY) scripts/uplift_models.py

site:
	$(PY) scripts/build_site.py

test:
	$(PY) -m pytest -q

deploy: site
	CLOUDFLARE_API_TOKEN="$$LEVELBROOK_CF_DEPLOY_TOKEN" CLOUDFLARE_ACCOUNT_ID=a67eceeb4b89d2d4171ed209e87c9456 \
	  npx wrangler pages deploy site --project-name=uplift-readout --branch=main --commit-dirty=true

clean:
	rm -f data/cupac.parquet data/nuisance.parquet results/*.json
