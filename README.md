# predict-online-game-behaviour-mlops

An MLOps project for mlops-zoomcamp.
It uses the Predict Online Gaming Behavior Dataset from Kaggle: https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset

## How to run?

### Orchestration (Mage AI), Experiment tracking (MLFlow)
To run the core of the project please run:

```bash
./scripts/start.sh
```
docker compose is required to run this command.
As a result a bunch of services will start, including Mage, Localstack, a Postgres db, Grafana, MLFlow and admirer.

The orchestration pipelines can be found on Mage under `localhost:6789`.
There is one pipeline for running the training and logging it to MLFlow and saving the model to s3 bucket.
And a second pipeline that selects the model with highest accuracy in the model registry and copies it to another s3 bucket, that is used for deployment.

Other services can ba accessed as well, please refer to `compose.yaml` for the correct ports.


### Deployment (Flask app)
To proceed with the deployment run the following code:
```bash
cd model_api
docker compose up
```

This requires the orchestration services already up, and the workflows in Mage AI already run (model available on s3).

You can test the server by running the `E2E_test.py` file, with the following commands:
```bash
pip install -r requirements.txt
python E2E_test.py
```

### Monitoring
A scripts that calculates a couple of metrics using `evidently` packages and saves them to Postgres db.
To run:

```bash
cd monitoring
pip install -r requirements.txt
PYTHONPATH=".." python monitoring.py
```

After the reports are saved in the db, a dashboard can be created in Grafana. An example was provided under `dashboards/dashboard.yaml`.

## Additional info
### pre-commit hook
This repo features a pre-commit hook for linting and formatting. To install it, please use:
```bash
pip install pre-commit
pre-commit install
```

### CI/CD - Github Actions job
There is a tiny Github Actions workflow that checks if the code is well formatted and linted. 

