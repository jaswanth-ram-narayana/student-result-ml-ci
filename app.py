
import time
import uuid
import logging
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, Response
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

app = Flask(__name__)

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

# Prometheus monitoring metrics
REQUEST_COUNT = Counter(
    "student_api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "student_api_request_duration_seconds",
    "API request processing time",
    ["method", "endpoint"]
)


# Request monitoring middleware
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
    duration = time.perf_counter() - request.start_time
    duration_ms = duration * 1000

    response.headers["X-Request-ID"] = request.request_id

    # Update Prometheus request counter
    REQUEST_COUNT.labels(
        request.method,
        request.path,
        str(response.status_code)
    ).inc()

    # Record request processing time
    REQUEST_LATENCY.labels(
        request.method,
        request.path
    ).observe(duration)

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


# Load trained ML model
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_result_model.pkl was not found."
        )

    return joblib.load(MODEL_PATH)


# Home endpoint
@app.get("/")
def home():
    logger.info("Home endpoint accessed")

    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


# Health monitoring endpoint
@app.get("/health")
def health_check():
    logger.info("Health check requested")

    return jsonify({
        "status": "healthy",
        "service": "student-result-prediction"
    })


# Prometheus metrics endpoint
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )


# Student result prediction endpoint
@app.post("/predict")
def predict():
    logger.info("Prediction request received")

    try:
        data = request.get_json(silent=True)

        if not data:
            logger.warning(
                "Prediction request has no JSON body"
            )

            return jsonify({
                "error": "JSON request body is required"
            }), 400

        missing_fields = [
            feature for feature in FEATURES
            if feature not in data
        ]

        if missing_fields:
            logger.warning(
                "Prediction request is missing fields"
            )

            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        model = load_model()

        prediction_code = int(
            model.predict(sample)[0]
        )

        prediction = (
            "PASS" if prediction_code == 1 else "FAIL"
        )

        logger.info(
            "Prediction completed successfully: %s",
            prediction
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except Exception:
        logger.exception(
            "Error occurred during prediction"
        )

        return jsonify({
            "error": "Internal server error"
        }), 500


# Run Flask application
if __name__ == "__main__":
    logger.info(
        "Starting Student Result Prediction API"
    )

    app.run(
        host="0.0.0.0",
        port=5000
    )