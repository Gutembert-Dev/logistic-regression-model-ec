"""
Tests for the flask app.
"""
# pylint: disable=E0401,C0103

# System imports
from unittest import TestCase

# Third-party imports
from explore_ai_demo.predictor import app


class TestApp(TestCase):
    """
    Tests for the API application's endpoints.
    """

    def setUp(self):
        """
        Setup a test app.
        """

    def test_predict(self):
        """
        Test that predict can be called with test data.
        """
        response = app.test_client().post(
            "/invocations",
            headers={"Content-type": "text/csv"},
            data="236838,01MAY2021,257,4,325,325,207,9,0,110,0,68,-4,0,0,0,0,0,1,0,0,0,1,1",
        )
        response_json = response.get_json()
        print(response_json)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response_json, {"predictions": "0.02333423877564179,0\n", "status": 200}
        )

    def test_predict_batch(self):
        """
        Test that predict can be called with batch test data.
        """
        response = app.test_client().post(
            "/invocations",
            headers={"Content-type": "text/csv"},
            data=f"""
                                               236838,01MAY2021,257,4,325,325,207,9,0,110,0,68,-4,0,0,0,0,0,1,0,0,0,1,1
                                               236838,01MAY2021,257,4,325,325,207,9,0,110,0,68,-4,0,0,0,0,0,1,0,0,0,1,1
                                               236838,01MAY2021,257,4,325,325,207,9,0,110,0,68,-4,0,0,0,0,0,1,0,0,0,1,1
                                               """,
        )
        response_json = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response_json,
            {
                # "predictions": "0.02333423877564179,0\n0.02333423877564179,0\n0.02333423877564179,0\n",
                "predictions": "0.02333423877564178,0\n0.02333423877564178,0\n0.02333423877564178,0\n",
                "status": 200,
            },
        )

    def test_predict_with_incorrect_request_body_format(self):
        """
        Test that predict can not be called without a request body.
        """
        response = app.test_client().post(
            "/invocations", headers={"Content-type": "application/json"}
        )
        response_json = response.get_json()
        self.assertEqual(response.status_code, 415)
        self.assertEqual(
            response_json,
            {
                "status": {
                    "code": 415,
                    "info": "Incorrect request body format",
                    "reason": "This predictor only supports CSV data",
                    "status": "FAILURE",
                }
            },
        )

    def test_predict_without_body(self):
        """
        Test that predict can not be called without a request body.
        """
        response = app.test_client().post(
            "/invocations", headers={"Content-type": "text/csv"}
        )
        response_json = response.get_json()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response_json,
            {
                "status": {
                    "code": 400,
                    "info": "Cannot find request body",
                    "reason": "Request body empty, invalid or has an incorrect header",
                    "status": "FAILURE",
                }
            },
        )

    def test_health_endpoint(self):
        """
        Test that the health check endpoint works.
        """

        response = app.test_client().get("/ping")
        response_json = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response_json, {"code": 200, "status": "SUCCESS"})
