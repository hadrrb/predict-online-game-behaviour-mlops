from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher
import pandas as pd

if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here

    enc = LabelEncoder()
    hasher = FeatureHasher(n_features=2, input_type="string")
    hashed_features = hasher.transform(df["Gender"].apply(lambda df: [df]))
    hashed_df = pd.DataFrame(
        hashed_features.toarray(), columns=[f"gender_{i}" for i in range(2)]
    )
    df = pd.concat([df, hashed_df], axis=1)
    df["Location"] = enc.fit_transform(df["Location"])
    df["GameGenre"] = enc.fit_transform(df["GameGenre"])
    df["GameDifficulty"] = enc.fit_transform(df["GameDifficulty"])
    df["PlayTimePerWeek"] = df["SessionsPerWeek"] * df["AvgSessionDurationMinutes"] / 60
    df = df.drop(
        ["SessionsPerWeek", "AvgSessionDurationMinutes", "Gender", "PlayerID"], axis=1
    )
    df["EngagementLevel"] = enc.fit_transform(df["EngagementLevel"])

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"
