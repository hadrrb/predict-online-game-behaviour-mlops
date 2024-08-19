import mlflow

if "custom" not in globals():
    from mage_ai.data_preparation.decorators import custom
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_custom(*args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here
    client = mlflow.deployments.get_deploy_client("local")

    # Load your model (replace with the correct path to your model)
    model_uri = "s3://game-behaviour-model/v1"  # or a local path, e.g., "path/to/model"

    client.create_deployment(name="game_behaviour", model_uri=model_uri)

    return {}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"
