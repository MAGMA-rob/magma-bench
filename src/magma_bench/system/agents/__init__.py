from .base_agent import Agent
from .docker_agent import DockerAgent
from .ollama_agent import (
    OllamaAgent,
    GPTOllamaAgent,
    QwenOllamaAgent,
    ManagedGPTOllamaAgent,
    ManagedQwenOllamaAgent,
    map_pipeline_role_to_chat_role,
)
