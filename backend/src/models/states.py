from typing import TypedDict, List, Dict, Optional
from langchain_core.messages import AnyMessage

class WorkflowState(TypedDict, total=False):
    # Required Inputs
    document_id: str

    # Processing Requirements
    validation_passed: bool
    progress_log: List[str]
    processing_config: Dict

    # Context and Messaging
    messages: Optional[List[AnyMessage]] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    image_data: Optional[List[Dict]] = None

    ## Common Inputs
    input_markdown: Optional[str] = []
    
    # Generic Result Objects
    markdown_result: Optional[str]
    json_result: Optional[Dict]
    markdown_name: Optional[str]
    json_name: Optional[str]
    result_path: Optional[str]

class DocumentProcessingState(WorkflowState):
    """
    AgentState for Document Processing and Conversion to Markdown
    """
    
    # Input
    tiff_path: str
    processing_context: str

    # Processing State
    pages: List[str]
    images: List[Dict]
    current_chunk: List[str]
    chunk_index: int
    total_chunks: int

    # Context and Messaging
    consolidation_context: Optional[str]
    chunk_results: List[str]

    # Output
    processing_path: str
