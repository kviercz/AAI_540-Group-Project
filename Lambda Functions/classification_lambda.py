import json
import boto3
import os
import cv2
import numpy as np

def lambda_handler(event, context):
    print("Received event:", json.dumps(event))

    # 1. Parse the S3 event
    record = event['Records'][0]
    s3_bucket = record['s3']['bucket']['name']
    s3_key = record['s3']['object']['key']
    print(f"New processed image: s3://{s3_bucket}/{s3_key}")

    # 2. Download the processed image locally
    s3 = boto3.client("s3")
    local_path = "/tmp/processed_image.png"  # or .jpg, depending on your file
    s3.download_file(s3_bucket, s3_key, local_path)

    # 3. Read and preprocess the image
    #    Adjust this step to match your model’s expected input
    image = cv2.imread(local_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to read image from {local_path}")
        return {"statusCode": 400, "body": "Image not found"}

    # Example: your model might expect 48x48 grayscale
    image = cv2.resize(image, (48, 48))
    image = image.astype("float32") / 255.0
    # Add batch dimension + channel dimension if needed
    image = np.expand_dims(image, axis=0)     # shape [1, 48, 48]
    image = np.expand_dims(image, axis=-1)    # shape [1, 48, 48, 1]

    # 4. Invoke the SageMaker endpoint
    endpoint_name = "FER-Image-Model-2025-03-02-1317"  # Your deployed endpoint
    runtime = boto3.client("sagemaker-runtime")

    payload = {"instances": image.tolist()}
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload)
    )

    # 5. Parse the prediction result
    result = json.loads(response['Body'].read().decode("utf-8"))
    print("Inference result:", result)

    # 6. (Optional) Store or log the result
    # For example, write a JSON with the prediction to an S3 folder
    result_bucket = s3_bucket
    result_key = s3_key.replace("processed/", "classification-results/") + ".json"
    s3.put_object(Bucket=result_bucket, Key=result_key, Body=json.dumps(result))
    print(f"Saved inference result to s3://{result_bucket}/{result_key}")

    return {
        "statusCode": 200,
        "body": json.dumps({"prediction": result})
    }
