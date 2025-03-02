import json
import boto3
import os
import time

def lambda_handler(event, context):
    print("Received event:", json.dumps(event))

    # 1. Parse S3 event data
    record = event['Records'][0]
    s3_bucket = record['s3']['bucket']['name']
    s3_key = record['s3']['object']['key']
    print(f"New file uploaded: s3://{s3_bucket}/{s3_key}")

    # 2. Construct unique job name
    job_name = f"preprocessing-job-{int(time.time())}"

    # 3. Initialize SageMaker client
    sagemaker_client = boto3.client("sagemaker")

    # 4. Create the processing job with updated ContainerEntrypoint to extract the tarball first
    response = sagemaker_client.create_processing_job(
        ProcessingJobName=job_name,
        RoleArn=os.environ["SAGEMAKER_ROLE_ARN"],  # Execution role for SageMaker
        AppSpecification={
            "ImageUri": os.environ["PROCESSING_IMAGE_URI"],
            "ContainerEntrypoint": [
                "bash",
                "-c",
                "tar -xzvf /opt/ml/processing/input/code/preprocessing.tar.gz -C /opt/ml/processing/input/code && python3 /opt/ml/processing/input/code/preprocessing.py"
            ]
        },
        ProcessingResources={
            "ClusterConfig": {
                "InstanceType": "ml.m5.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30
            }
        },
        ProcessingInputs=[
            {
                "InputName": "script-code",
                "S3Input": {
                    "S3Uri": f"s3://{s3_bucket}/code/preprocessing.tar.gz",
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File"
                }
            },
            {
                "InputName": "input-data",
                "S3Input": {
                    "S3Uri": f"s3://{s3_bucket}/{s3_key}",
                    "LocalPath": "/opt/ml/processing/input/data",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File"
                }
            }
        ],
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "output-data",
                    "S3Output": {
                        "S3Uri": f"s3://{s3_bucket}/processed/",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob"
                    }
                }
            ]
        }
    )

    print(f"Started SageMaker Processing Job: {job_name}")
    return {
        "statusCode": 200,
        "body": json.dumps(f"Processing job {job_name} started.")
    }
