import boto3
import json
import base64
import streamlit as st
import speech_recognition as spr
r = spr.Recognizer()

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name='us-west-2'
)

def generate_images(input_text):
    st.sidebar.success("Please wait for result...")
    payload = {
        "modelId": "amazon.titan-image-generator-v1",
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps({
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": input_text, "negativeText": "red dot"},
            "imageGenerationConfig": {
                "numberOfImages": 3,
                "quality": "standard",
                "height": 1024,
                "width": 1024,
                "cfgScale": 8.0,
                "seed": 0
            }
        })
    }

    response = bedrock_runtime.invoke_model(
        body=payload["body"],
        contentType=payload["contentType"],
        accept=payload["accept"],
        modelId=payload["modelId"]
    )

    data = json.loads(response['body'].read())

    num_images = 3
    images_per_column = 1

    columns = st.columns(num_images // images_per_column)

    i = 0
    for indx, image in enumerate(data["images"]):
        with columns[i % (num_images // images_per_column)]:
            st.image(base64.b64decode(image), caption=f"Image {i}", use_column_width=True)
        i += 1

radio = st.sidebar.radio("Pick one", ["Input_Text","Speech_input"])
# st.sidebar.warning(radio)
if radio == 'Input_Text':
    input_text = st.sidebar.text_input("Enter text for image generation:", placeholder="write something..")
    if st.sidebar.button("Generate Images"):
        if input_text:
            generate_images(input_text)
else:
    speech_input = st.sidebar.button("Use Speech Input")

    info_text = st.sidebar.empty()
    if speech_input:
        with spr.Microphone() as source:
            info_text.write("Say something...")
            r.adjust_for_ambient_noise(source)
            audio = r.record(source, duration=3)
            try:
                input_text = r.recognize_google(audio)
                info_text.success(f"You said: {input_text}")
                generate_images(input_text)
            except spr.UnknownValueError:
                info_text.warning("Sorry, could not understand audio.")
            except spr.RequestError as e:
                info_text.error(f"Speech recognition request failed: {e}")

