.PHONY: install data finetune embed heads sweep-data sweep-circuit test serve

install:
	pip install -r requirements.txt

data:
	python -m qpd.data

finetune:
	python -m qpd.train stage=finetune

embed:
	python -m qpd.train stage=embed

heads:
	python -m qpd.train stage=head --multirun head=logreg,mlp,quantum seed=0,1,2

sweep-data:
	python -m qpd.train stage=head --multirun head=mlp,quantum data_frac=0.01,0.05,0.25,1.0 seed=0,1,2

sweep-circuit:
	python -m qpd.train stage=head --multirun head=mlp,quantum quantum.n_qubits=4,6,8 quantum.n_layers=1,2,4 seed=0,1,2

test:
	pytest -q

serve:
	uvicorn app.main:app --reload
