import boto3
import json
import base64
import streamlit as st

bedrock_runtime = boto3.client(
    service_name = "bedrock-runtime",
    region_name = 'us-west-2'
)
input = 'green lion with goats'

payload = {
  "modelId": "amazon.titan-image-generator-v1",
  "contentType": "application/json",
  "accept": "application/json",
  "body": "{\"taskType\": \"TEXT_IMAGE\",\"textToImageParams\": {\"text\":\""+input+"\",\"negativeText\":\"red dot\"}, \"imageGenerationConfig\": {\"numberOfImages\": 3,\"quality\": \"standard\",\"height\": 1024,\"width\": 1024,\"cfgScale\": 8.0,\"seed\": 0}}" 
}

response = bedrock_runtime.invoke_model(
    body = payload["body"],
    contentType = payload["contentType"],
    accept =  payload["accept"],
    modelId = payload["modelId"]
)

data = json.loads(response['body'].read())

i=0
for image in enumerate(data["images"]):
    with open(f"./Amzout/{i} {input[0:5]}.png", "wb") as f:
        f.write(base64.b64decode(image[1]))
    st.image(f"./Amzout/{i} {input[0:5]}.png")
    i=i+1