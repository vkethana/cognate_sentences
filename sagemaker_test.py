import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
import os

HFTOKEN = os.environ.get('HFTOKEN')

try:
	role = sagemaker.get_execution_role()
except ValueError:
	iam = boto3.client('iam')
	role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

# Hub Model configuration. https://huggingface.co/models
hub = {
	'HF_MODEL_ID':'mistralai/Mixtral-8x7B-v0.1',
	'SM_NUM_GPUS': json.dumps(8),
	'HUGGING_FACE_HUB_TOKEN': HFTOKEN
}

assert hub[TOKEN] != '<REPLACE WITH YOUR TOKEN>', "You have to provide a token."

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
	image_uri=get_huggingface_llm_image_uri("huggingface",version="2.0.2"),
	env=hub,
	role=role
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
	initial_instance_count=1,
	instance_type="ml.g5.48xlarge",
	container_startup_health_check_timeout=300,
  )
  
# send request
predictor.predict({
	"inputs": "Mon nom est Julien et j'aime",
})
