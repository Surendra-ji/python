from platform import python_version
tested_version = "3.10."
version = python_version()
print(f"You are using Python {version}")
if not version.startswith(tested_version):
    print(f"This notebook was tested with {tested_version}")
    print("Some parts might behave unexpectedly with a different Python version")

import sagemaker, boto3, json
from sagemaker.session import Session

sagemaker_session = Session()
aws_role = sagemaker_session.get_caller_identity_arn()
aws_region = boto3.Session().region_name
sess = sagemaker.Session()

model_id = "huggingface-text2text-flan-t5-large"
model_version = "1.*"

from ipywidgets import widgets
from sagemaker.jumpstart.notebook_utils import list_jumpstart_models

# Retrieves all Text2Text Generation models available by SageMaker Built-In Algorithms.
filter_value = "task == text2text"
text2text_generation_models = list_jumpstart_models(filter=filter_value)

# display the model-ids in a dropdown to select a model for inference.
model_dropdown = widgets.Dropdown(
    options=text2text_generation_models,
    value=model_id,
    description="Select a model",
    style={"description_width": "initial"},
    layout={"width": "max-content"},
)

display(model_dropdown)

model_id, model_version = model_dropdown.value, "1.*"

from sagemaker import image_uris, model_uris
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.predictor import Predictor

inference_instance_type = "ml.p3.8xlarge"

# Note that larger instances, e.g., "ml.g5.12xlarge" might be required for larger models,
# such as huggingface-text2text-flan-t5-xxl or huggingface-text2text-flan-ul2-bf16
# However, at present ml.g5.* instances are not supported in batch transforms.
# Thus, if using such an instance, please skip Sections 6 and 7 of this notebook.

# Retrieve the inference docker container uri. This is the base HuggingFace container image for the default model above.
deploy_image_uri = image_uris.retrieve(
    region=None,
    framework=None,  # automatically inferred from model_id
    image_scope="inference",
    model_id=model_id,
    model_version=model_version,
    instance_type=inference_instance_type,
)

# Retrieve the model uri.
model_uri = model_uris.retrieve(
    model_id=model_id, model_version=model_version, model_scope="inference"
)

model = JumpStartModel(
    model_id=model_id,
    image_uri=deploy_image_uri,
    model_data=model_uri,
    role=aws_role,
    predictor_cls=Predictor,
)

# Specify the batch job hyperparameters here, If you want to treate each example hyperparameters different please pass hyper_params_dict as None
hyper_params = {"max_length": 30, "top_k": 50, "top_p": 0.95, "do_sample": True}
hyper_params_dict = {"HYPER_PARAMS": str(hyper_params)}

# We will use the cnn_dailymail dataset from HuggingFace over here
from datasets import load_dataset

cnn_test = load_dataset("cnn_dailymail", "3.0.0", split="test")
# Choosing a smaller dataset for demo purposes. You can use the complete dataset as well.
cnn_test = cnn_test.select(list(range(20)))

# We will use a default s3 bucket for providing the input and output paths for batch transform
output_bucket = sess.default_bucket()
output_prefix = "jumpstart-example-text2text-batch-transform"

s3_input_data_path = f"s3://{output_bucket}/{output_prefix}/batch_input/"
s3_output_data_path = f"s3://{output_bucket}/{output_prefix}/batch_output/"

# You can specify a prompt here
prompt = "Briefly summarize this text: "

import json
import boto3
import os

# Provide the test data and the ground truth file name
test_data_file_name = "articles.jsonl"
test_reference_file_name = "highlights.jsonl"

test_articles = []
test_highlights = []

# We will go over each data entry and create the data in the input required format as described above
for i, test_entry in enumerate(cnn_test):
    article = test_entry["article"]
    highlights = test_entry["highlights"]
    # Create a payload like this if you want to have different hyperparameters for each test input
    # payload = {"id": id,"text_inputs": f"{prompt}{article}", "max_length": 100, "temperature": 0.95}
    # Note that if you specify hyperparameter for each payload individually,
    # you may want to ensure that hyper_params_dict is set to None instead
    payload = {"id": i, "text_inputs": f"{prompt}{article}"}
    test_articles.append(payload)
    test_highlights.append({"id": i, "highlights": highlights})

with open(test_data_file_name, "w") as outfile:
    for entry in test_articles:
        outfile.write(f"{json.dumps(entry)}\n")

with open(test_reference_file_name, "w") as outfile:
    for entry in test_highlights:
        outfile.write(f"{json.dumps(entry)}\n")

# Uploading the data
s3 = boto3.client("s3")
s3.upload_file(test_data_file_name, output_bucket, f"{output_prefix}/batch_input/articles.jsonl")
# Creating the batch transformer object. If you have a large dataset you can
# divide it into smaller chunks and use more instances for faster inference
batch_transformer = model.transformer(
    instance_count=1,
    instance_type=inference_instance_type,
    output_path=s3_output_data_path,
    assemble_with="Line",
    accept="text/csv",
    max_payload=1,
)
batch_transformer.env = hyper_params_dict

# Making the predictions on the input data
batch_transformer.transform(
    s3_input_data_path, content_type="application/jsonlines", split_type="Line"
)

batch_transformer.wait()

import ast
import evaluate
import pandas as pd

# Downloading the predictions
s3.download_file(
    output_bucket, output_prefix + "/batch_output/" + "articles.jsonl.out", "predict.jsonl"
)

with open("predict.jsonl", "r") as json_file:
    json_list = list(json_file)

# Creating the prediction list for the dataframe
predict_dict_list = []
for predict in json_list:
    if len(predict) > 1:
        predict_dict = ast.literal_eval(predict)
        predict_dict_req = {
            "id": predict_dict["id"],
            "prediction": predict_dict["generated_texts"][0],
        }
        predict_dict_list.append(predict_dict_req)

# Creating the predictions dataframe
predict_df = pd.DataFrame(predict_dict_list)

test_highlights_df = pd.DataFrame(test_highlights)

# Combining the predict dataframe with the original summarization on id to compute the rouge score
df_merge = test_highlights_df.merge(predict_df, on="id", how="left")

rouge = evaluate.load("rouge")
results = rouge.compute(
    predictions=list(df_merge["prediction"]), references=list(df_merge["highlights"])
)
print(results)

## Delete the SageMaker model
batch_transformer.delete_model()

from sagemaker.utils import name_from_base

endpoint_name = name_from_base(f"jumpstart-example-{model_id}")
# Deploy the Model. Note that we need to pass Predictor class when we deploy model through Model class,
# for being able to run inference through the Sagemaker API.
model_predictor = model.deploy(
    initial_instance_count=1,
    instance_type=inference_instance_type,
    endpoint_name=endpoint_name,
)

# Provide all the text inputs to the model as a list
text_inputs = [entry["text_inputs"] for entry in test_articles[:10]]
# The information about the different parameters is provided above
payload = {
    "text_inputs": text_inputs,
    "max_length": 30,
    "num_return_sequences": 1,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": True,
}


def query_endpoint_with_json_payload(encoded_json, endpoint_name):
    client = boto3.client("runtime.sagemaker")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=encoded_json
    )
    return response


def parse_response_multiple_texts(query_response):
    model_predictions = json.loads(query_response["Body"].read())
    return model_predictions


query_response = query_endpoint_with_json_payload(
    json.dumps(payload).encode("utf-8"), endpoint_name=endpoint_name
)
generated_text_list = parse_response_multiple_texts(query_response)
print(*generated_text_list, sep="\n")
# Delete the SageMaker endpoint
model_predictor.delete_model()
model_predictor.delete_endpoint()