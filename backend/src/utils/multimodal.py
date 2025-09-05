from typing import List, Dict, Optional, Callable
from langchain_core.messages import HumanMessage
import base64
import logging

logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path: str) -> str:
    """"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_image_content(
        img_data: Dict, 
        text: str = "", 
        detail_level: str = "low"
        ) -> List[Dict]:
    """
    Create image content for a single image with optional text
    """
    content = []
    if text:
        content.append({"type": "text", "text": text})
    content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img_data['mime_type']};base64,{img_data['content']}",
                    "detail": detail_level
                }
            })
    return content

def create_batch_image_content(
        image_data: List[Dict],
        detail_level: str = "low",
        text_template: str = "Image {index} of {total}: ",
        intro_text: Optional[str] = None
) -> List[Dict]:
    """Create a *single* HumanMessage that contains many images."""
    content: List[Dict] = []

    if intro_text:
        content.append({"type": "text", "text": intro_text})
    for i, img_data in enumerate(image_data, 1):
        text = text_template.format(index=i, total=len(image_data))
        content.extend(create_image_content(img_data, text, detail_level))

    return content

def create_individual_image_messages(
        image_data: List[Dict],
        detail_level: str = "low",
        text_template: str = "Image {index} of {total}: ",
        text_generator: Optional[Callable[[int, Dict], str]] = None
) -> List[HumanMessage]:
    """Create a separate HumanMessage for each image."""
    messages: List[HumanMessage] = []

    for i, img_data in enumerate(image_data, 1):
        if text_generator:
            text = text_generator(i, img_data)
        else:
            text = text_template.format(index=i, total=len(image_data))

        #   ^^^^^^^^  PASS **img_data**, not the whole list!
        content = create_image_content(img_data, text, detail_level)
        messages.append(HumanMessage(content=content))

    return messages

def transform_to_multimodal(
        intro_text: str,
        image_data: List[Dict],
        detail_level: str = "low",
        mode: str = "individual", # "individual", "batch"
        text_template: str = "Image {index} of {total}: ",
        text_generator: Optional[Callable[[int, Dict], str]] = None
        ) -> List:
    """
    Transform user message with documents to multimodal format.

    Args:
        messages: List of messages to process
        document_contents: Dictionary of Base64 encoded document contents
        detail_level: Image detail level for processing

    Returns:
        Tuple of (transformed_messages, estimated_word_count)
    """
    if not image_data:
        return [HumanMessage(content=intro_text)] if intro_text else []
    
    messages = []

    if intro_text:
        messages.append(HumanMessage(content=intro_text))

    if mode == "batch":
        content = create_batch_image_content(image_data, detail_level, text_template)
        messages.append(HumanMessage(content=content))
    else:
        image_messages = create_individual_image_messages(
            image_data, detail_level, text_template, text_generator
        )
        messages.extend(image_messages)
    
    return messages