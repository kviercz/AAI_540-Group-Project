import json
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = "/opt/ml/model/fer_best_model.h5"  # SageMaker extracts model here

def model_fn(model_dir):
    """Load model for inference"""
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def input_fn(request_body, request_content_type):
    """Preprocess input request"""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["instances"])  # Expecting {"instances": [[...]]} format
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Perform inference"""
    predictions = model.predict(input_data)
    return predictions.tolist()

def output_fn(prediction, response_content_type):
    """Format response"""
    return json.dumps({"predictions": prediction})


