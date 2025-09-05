from abc import ABC, abstractmethod
from typing import List, Dict
from langchain_core.messages import HumanMessage
from utils.multimodal import transform_to_multimodal

class MessageBuildStrategy(ABC):
    """
    Base Strategy for building multimodal messages
    """
    @abstractmethod
    def build_messages(self, image_data: List[Dict], detail_level: str = "low", context: str = "") -> List[HumanMessage]:
        """
        Build appropriate message sequence for the use case
        """
        pass

class DocumentPagesStrategy(MessageBuildStrategy):
    """
    Strategy for multi-page document processing
    """
    def build_messages(self, image_data: List[Dict], detail_level: str = "low", context: str = "") -> List[HumanMessage]:
        intro = context or "Process these files which are pages of a larger file"

        return transform_to_multimodal(
            intro_text=intro,
            image_data=image_data,
            detail_level=detail_level,
            mode="individual",
            text_template="This is page {index} of {total} of the larger file"
        )
    
class SingleImageStrategy(MessageBuildStrategy):
    """
    Strategy for single image analysis
    """

    def build_messages(self, image_data: List[Dict], detail_level: str = "low", context: str = "") -> List[HumanMessage]:
        intro = context or "Analyze this image"

        return transform_to_multimodal(
            intro_text=intro,
            image_data=image_data[:1],
            detail_level=detail_level,
            mode="individual",
            text_template=""
        )
    
class BatchComparisonStrategy(MessageBuildStrategy):
    """
    Strategy for comparing multiple images in one message
    """
    def build_messages(self, image_data: List[Dict], detail_level: str = "low", context: str = "") -> List[HumanMessage]:
        intro = context or "Analyze and compare these images"

        return transform_to_multimodal(
            intro_text=intro,
            image_data=image_data,
            detail_level=detail_level,
            mode="batch",
            text_template="Image {index}:"
        )