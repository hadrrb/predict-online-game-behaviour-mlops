import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

if "data_loader" not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs):
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your data loading logic here
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("game-behaviour")
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=[
            "metrics.accuracy DESC",
            "metrics.precision DESC",
            "metrics.recall DESC",
        ],
    )[0]

    return f"{experiment.experiment_id}/{best_run.info.run_id}"


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"
