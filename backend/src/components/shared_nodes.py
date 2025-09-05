from typing import Any, Dict
import json
from pathlib import Path
from models.states import *
from datetime import datetime, timezone

class SharedNodes:
    """
    """
    def __init__(self):
        pass
    
    def validate_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reusable input validate for all graphs
        """
        if "document_id" not in state:
            raise ValueError("document_id required in state")
        
        state["validation_passed"] = True
        return state
    
    def load_document_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load markdown content from results folder
        """
        document_id = state["document_id"]
        markdown_path = f"{state["result_path"]}/{document_id}/document.md"

        if Path(markdown_path).exists():
            with open(markdown_path, 'r') as f:
                state["input_markdown"] = f.read()
        else:
            raise FileNotFoundError(f"Markdown not found for {document_id}")

        return state

    def log_progress(self, state: Dict[str, Any], message) -> Dict[str, Any]:
        """
        Log progress on the graph
        """
        if "progress_log" not in state:
            state["progress_log"] = []
        
        state["progress_log"].append({
            "timestamp": datetime.now(timezone.utc),
            "message": message
        })

        return state