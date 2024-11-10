import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO
import json
import ollama
from icon import page_icon


st.set_page_config(
    page_title="LLaVA Playground",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)


def img_to_base64(image):
    """
    Convert an image to base64 format.

    Args:
        image: PIL.Image - The image to be converted.
    Returns:
        str: The base64 encoded image.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def get_allowed_model_names(models_info: dict) -> tuple:
    """
    Returns a tuple containing the names of the allowed models.
    """
    allowed_models = ["bakllava:latest", "llava:latest"]
    return tuple(
        model
        for model in allowed_models
        if model in [m["name"] for m in models_info["models"]]
    )