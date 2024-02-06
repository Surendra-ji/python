from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
import boto3
import os
import streamlit as st

os.environ["AWS_PROFILE"] = "default"

#bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# modelID = "cohere.command-text-v14" 
# modelID = "ai21.j2-ultra-v1"
modelID = "anthropic.claude-v2:1"
# modelID = "stability.stable-diffusion-xl-v0"



llm = Bedrock(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={"max_tokens_to_sample": 2000,"temperature":0.9} #for anthropic use max_tokens_to_sample and for cohere use max_tokens
    # {"text_prompts": [{"text":"this is where you place your input text"}],"cfg_scale":10,"seed":0,"steps":50}
)

def my_chatbot(language,freeform_text):
    prompt = PromptTemplate(
        input_variables=["language", "freeform_text"],
        template="You are a chatbot. You are in {language}.\n\n{freeform_text}"
    )

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)

    response=bedrock_chain({'language':language, 'freeform_text':freeform_text})
    return response

st.title("Bedrock Chatbot with model: " + modelID)

language = st.sidebar.selectbox("Language", ["hindi", "english", "Punjabi"])

if language:
    freeform_text = st.sidebar.text_area(label="what is your question?",
    max_chars=100)

if freeform_text:
    response = my_chatbot(language,freeform_text)
    st.write(response['text'])