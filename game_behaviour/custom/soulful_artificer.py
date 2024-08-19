import boto3

if "custom" not in globals():
    from mage_ai.data_preparation.decorators import custom
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_custom(best_model, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here
    s3 = boto3.client("s3", endpoint_url="http://localstack:4566")

    source_bucket = "predict-game-behaviour"
    source_prefix = f"{best_model}/artifacts/rdc/"  # Make sure to end with '/'

    destination_bucket = "game-behaviour-model"
    destination_prefix = "v1/"  # Make sure to end with '/'

    # List objects within the source directory
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=source_bucket, Prefix=source_prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                source_key = obj["Key"]
                destination_key = source_key.replace(
                    source_prefix, destination_prefix, 1
                )

                copy_source = {"Bucket": source_bucket, "Key": source_key}

                s3.copy(
                    CopySource=copy_source,
                    Bucket=destination_bucket,
                    Key=destination_key,
                )
                print(f"Copied {source_key} to {destination_key}")
    return {}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"
