#!/bin/bash

pip install boto3
mlflow server --host 0.0.0.0 --port 5000 --artifacts-destination s3://predict-game-behaviour/