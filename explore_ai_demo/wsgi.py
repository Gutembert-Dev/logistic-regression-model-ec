"""
This is just a simple wrapper for gunicorn to find your app.
If you want to change the algorithm file, simply change "predictor" above to the new file.
"""
# pylint: disable=E0401
from explore_ai_demo.predictor import app

app = app.app
