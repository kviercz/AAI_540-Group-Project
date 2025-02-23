import argparse
import boto3
import sagemaker
from datetime import datetime
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.model import Model

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_s3_path", type=str, required=True)  # Model artifact location in S3
parser.add_argument("--s3_capture_upload_path", type=str, required=True)  # Data capture path in S3
parser.add_argument("--region", type=str, required=True) # Region
args = parser.parse_args()

# Define Model Artifact & Data Capture Path
model_s3_path = args.model_s3_path
s3_capture_upload_path = args.s3_capture_upload_path
region = args.region

# Setup session
boto3.setup_default_session(region_name=region)
sagemaker_session = sagemaker.Session()

# Generate Endpoint Name
endpoint_name = f"FER-Image-Model-{datetime.utcnow():%Y-%m-%d-%H%M}"
print(f"Deploying model to endpoint: {endpoint_name}")

# Initialize SageMaker Session & Role
# sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Configure Data Capture
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri=s3_capture_upload_path
)

# Define the Model for Deployment
model = Model(
    model_data=model_s3_path,
    role=role,
    sagemaker_session=sagemaker_session
)

# Deploy the Model to an Endpoint
model.deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge", 
    endpoint_name=endpoint_name,
    data_capture_config=data_capture_config
)

print(f"Model successfully deployed to endpoint: {endpoint_name}")
