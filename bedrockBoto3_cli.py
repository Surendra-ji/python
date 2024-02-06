import boto3
import json

bedrock = boto3.client(
    service_name = "bedrock",
    region_name = 'us-west-2'
)

bedrock_runtime = boto3.client(
    service_name = "bedrock-runtime",
    region_name = 'us-west-2'
)

prompt = 'Wish Me Happy New Year 2024'

payload = {
  "modelId": "cohere.command-text-v14",
  "contentType": "application/json",
  "accept": "*/*",
  "body": json.dumps({"prompt":prompt, "max_tokens":100, "temperature":0.8})
  #"{\"prompt\":\""+ prompt+"\", \"max_tokens\":100, \"temperature\":0.8}"
}

response = bedrock_runtime.invoke_model(
    body = payload["body"],
    contentType = "application/json",
    accept =  "*/*",
    modelId = payload["modelId"]
)

data = json.loads(response['body'].read())
print(data.get("generations")[0]['text'])