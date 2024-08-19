import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

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
    y = df["EngagementLevel"]
    X = df.drop(["EngagementLevel"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("game-behaviour")

    with mlflow.start_run():
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="rdc",
            input_example=X_train,
            registered_model_name="sklearn-random-forest-model",
        )
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric(
            "precision", precision_score(y_test, y_pred, average="weighted")
        )
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average="weighted"))

    return clf


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"
