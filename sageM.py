from typing import NamedTuple, Dict, Any
import sagemaker
from sagemaker.jumpstart.model import JumpStartModel

class JumpStartChatbotModelConfig(NamedTuple):
    model_id: str
    model_kwargs: Dict[str, Any] = {}
    payload_kwargs: Dict[str, Any] = {}

jumpstart_chatbot_models_config = [
    JumpStartChatbotModelConfig(
        model_id="huggingface-llm-falcon-7b-instruct-bf16",
        payload_kwargs={"return_full_text": True},
    ),
    JumpStartChatbotModelConfig(
        model_id="huggingface-textgeneration-falcon-40b-instruct-bf16",
        payload_kwargs={"return_full_text": True},
    ),
    JumpStartChatbotModelConfig(
        model_id="huggingface-textgeneration1-redpajama-incite-chat-3B-v1-fp16",
    ),
    JumpStartChatbotModelConfig(
        model_id="huggingface-textgeneration1-redpajama-incite-chat-7B-v1-fp16",
    ),
    JumpStartChatbotModelConfig(
        model_id="huggingface-textgeneration2-gpt-neoxt-chat-base-20b-fp16",
    ),
]

# Set the desired model ID
selected_model_id = "huggingface-llm-falcon-7b-instruct-bf16"

# Find the corresponding model configuration
selected_model_config = next(
    config for config in jumpstart_chatbot_models_config if config.model_id == selected_model_id
)

# Display the selected model information
print("Selected Model: ", selected_model_config.model_id)
role = 'arn:aws:iam::459425521263:role/aws-service-role'
print("role: ",role)
model = JumpStartModel(
    model_id=selected_model_config.model_id,
    model_version="1.3.2",
    role=role,
)
predictor = model.deploy()
print("predictor: ",predictor)
