from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Define the S3 URI for the model
model_uri = "s3://game-behaviour-model/v1/"

# Load the model from S3
model = mlflow.pyfunc.load_model(model_uri)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the JSON request data
        data = request.json

        # Convert the input data into a DataFrame
        print(data)
        input_df = pd.DataFrame(data)

        # Make predictions
        predictions = model.predict(input_df)

        # Convert the predictions to a list and return as JSON
        return predictions.tolist()

    except Exception as e:
        # If an error occurs, return the error message
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    # Run the Flask application
    app.run(host="0.0.0.0", port=7777)
