import time
import uuid
logger = logging.getLogger(__name__)
@app.before_request
def start_request_logging():
    request.request_id = request.headers.get(
        "X-Request-ID", str(uuid.uuid4())
    )
    request.start_time = time.perf_counter()

    logger.info(
        "event=request_started request_id=%s method=%s path=%s",
        request.request_id,
        request.method,
        request.path
    )


@app.after_request
def log_request_response(response):
    duration_ms = (
        time.perf_counter() - request.start_time
    ) * 1000

    response.headers["X-Request-ID"] = request.request_id

    log_level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        log_level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request.request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    return response
from pathlib import Path
import logging

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_result_model.pkl was not found."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def home():
    logger.info("Home endpoint accessed")
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


@app.get("/health")
def health_check():
    logger.info("Health check requested")
    return jsonify({
        "status": "healthy",
        "service": "student-result-prediction"
    })


@app.post("/predict")
def predict():
    logger.info("Prediction request received")

    try:
        data = request.get_json(silent=True)

        if not data:
            logger.warning("Prediction request has no JSON body")
            return jsonify({
                "error": "JSON request body is required"
            }), 400

        missing_fields = [
            feature for feature in FEATURES
            if feature not in data
        ]

        if missing_fields:
            logger.warning("Prediction request is missing fields")
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        model = load_model()
        prediction_code = int(model.predict(sample)[0])
        prediction = "PASS" if prediction_code == 1 else "FAIL"

        logger.info("Prediction completed successfully: %s", prediction)

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except Exception:
        logger.exception("Error occurred during prediction")
        return jsonify({
            "error": "Internal server error"
        }), 500


if __name__ == "__main__":
    logger.info("Starting Student Result Prediction API")
    app.run(host="0.0.0.0", port=5000)
