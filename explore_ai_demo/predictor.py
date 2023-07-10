"""
Flask API entry point.

This file also contains the definition of the flask
application that handles Http requests.
"""
# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.
# pylint: disable=E0401
from __future__ import print_function

import io
import flask
import pandas
import logging

from explore_ai_demo.model import P2PModel

from explore_ai_demo.constants import FEATURES_COLUMNS, IDENTIFIER_COLUMNS

model = P2PModel()
# Load the model
model.load()

# The flask a--platform=linux/amd64pp for serving predictions
app = flask.Flask(__name__)

logging.basicConfig(level=logging.DEBUG)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = model is not None  # You can insert a health check here
    #
    status = 200 if health else 404
    return flask.jsonify({"code": status, "status": "SUCCESS"})


@app.route("/invocations", methods=["POST"])
def predict():
    """
    Handle a request to predict endpoint.
    """

    if flask.request.content_type == "text/csv":
        try:
            data = flask.request.data.decode("utf-8")
            print(data)
            output = io.StringIO(data)
            logging.info(
                f"""
            Output {output}
            """
            )
            data = pandas.read_csv(output, header=None)
            logging.info(
                f"""
                Output of data
                {data}
            """
            )
            data.columns = IDENTIFIER_COLUMNS + FEATURES_COLUMNS

            data = data[FEATURES_COLUMNS]
            logging.info(
                f"""
                data.columns
                {data.columns}
            """
            )
        # pylint: disable=broad-except
        except Exception:
            data = None
        if data is None:  # check for None when empty request is sent
            return (
                flask.jsonify(
                    {
                        "status": {
                            "code": 400,
                            "info": "Cannot find request body",
                            "reason": "Request body empty, invalid or has an incorrect header",
                            "status": "FAILURE",
                        }
                    }
                ),
                400,
            )

    else:
        return (
            flask.jsonify(
                {
                    "status": {
                        "code": 415,
                        "info": "Incorrect request body format",
                        "reason": "This predictor only supports CSV data",
                        "status": "FAILURE",
                    }
                }
            ),
            415,
        )
    logging.info(f"Invoked with {data.shape} records")

    # Do the prediction
    predictions = model.predict(data)

    # Convert from numpy back to CSV
    out = io.StringIO()
    pandas.DataFrame(predictions).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.jsonify({"predictions": result, "status": 200})
