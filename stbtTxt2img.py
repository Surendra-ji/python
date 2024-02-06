import boto3
import json
import base64

bedrock_runtime = boto3.client(
    service_name = "bedrock-runtime",
    region_name = 'us-west-2'
)
input = 'kachcha badam'

payload = {
  "modelId": "stability.stable-diffusion-xl-v0",
  "contentType": "application/json",
  "accept": "*/*",
  "body": "{\"text_prompts\": [{\"text\":\""+input+"\"}],\"cfg_scale\":10,\"seed\":0,\"steps\":50}"  
}

response = bedrock_runtime.invoke_model(
    body = payload["body"],
    contentType = payload["contentType"],
    accept =  payload["accept"],
    modelId = payload["modelId"]
)

data = json.loads(response['body'].read())
for i, image in enumerate(data["artifacts"]):
    with open(f"./stabilityImg/{input[0:4]}{i}.png", "wb") as f:
        f.write(base64.b64decode(image["base64"]))