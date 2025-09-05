# document_processing_graph.py
from pathlib import Path
from typing import List, Dict, Any
import math
import logging
from langgraph.graph import StateGraph, START, END
from langchain.schema import HumanMessage, SystemMessage

from graphs.base_graph import BaseDocumentGraph
from components.assistants import MultiModalAsssitant
from components.message_strategies import DocumentPagesStrategy
from services.tiff_processor import TiffProcessor
from models.states import DocumentProcessingState  # <-- Pydantic model

# ------------------------------------------------------------------
# 1.  Configure logger (optional but handy)
# ------------------------------------------------------------------
logger = logging.getLogger(__name__)

class DocumentProcessingGraph(BaseDocumentGraph[DocumentProcessingState]):
    """
    Graph that validates, converts TIFF → PNG, runs the assistant,
    extracts Markdown, and saves the result.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        verbose: bool,
        logprobs: bool,
        reasoning_effort: str,
        max_tokens: int,
    ):
        # Heavy‑weight objects must exist *before* the base class builds the graph.
        self.tiff_processor = TiffProcessor()
        self.message_strategy = DocumentPagesStrategy()
        self.assistant = MultiModalAsssitant(
            message_strategy=self.message_strategy,
            provider=provider,
            model=model,
            tools=None,
            verbose=verbose,
            logprobs=logprobs,
            streaming=False,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        super().__init__(state_schema=DocumentProcessingState)

    # ------------------------------------------------------------------
    # 2.  Build the graph
    # ------------------------------------------------------------------
    def build_graph(self) -> StateGraph[DocumentProcessingState]:
        graph = StateGraph(state_schema=self.state_schema)
        # Attach the common nodes (validate_input & save_results)
        self.add_common_nodes(graph)

        # ---- Node definitions -----------------------------------------
        graph.add_node("convert_tiff", self.convert_tiff_to_pages)
        graph.add_node("prepare_processing", self.prepare_for_processing)
        graph.add_node("chunk_and_process", self.chunk_and_process)
        graph.add_node("consolidate_markdown", self.consolidate_markdown)

        # ---- Flow wiring -----------------------------------------------
        graph.add_edge(START, "validate_input")
        graph.add_edge("validate_input", "convert_tiff")
        graph.add_edge("convert_tiff", "prepare_processing")
        graph.add_edge("prepare_processing", "chunk_and_process")
        graph.add_edge("chunk_and_process", "consolidate_markdown")
        graph.add_edge("consolidate_markdown", "save_results")  # fixed typo
        graph.add_edge("save_results", END)

        return graph

    # ------------------------------------------------------------------
    # 3. Node implementations
    # ------------------------------------------------------------------
    def convert_tiff_to_pages(
        self,
        state: dict,   # the state is a plain dict - the type hint can stay
    ) -> dict:
        """
        Convert a TIFF file to PNG pages and extract metadata.
        """
        tiff_path = state.get("tiff_path")
        if not tiff_path:
            raise ValueError("Missing 'tiff_path' in state")

        logger.info(f"Converting TIFF: {tiff_path}")
        try:
            result = self.tiff_processor.process_tiff(tiff_path)

            # Expected structure:
            #   {"pages": [...], "image_data": [...], "metadata": {...}}
            state["pages"] = result["pages"]
            state["images"] = result["image_data"]
            state["json_result"] = result["metadata"]
            return state
        except Exception as e:
            document_id = state.get("document_id", "unknown")
            logger.error(f"Failed to convert TIFF for document {document_id}: {e}")
            raise


    def prepare_for_processing(
        self,
        state: dict,
    ) -> dict:
        """
        Mutate the state before the assistant runs.
        """
        tiff_path = state.get("tiff_path")
        if not tiff_path:
            raise ValueError("Missing 'tiff_path' in state")

        document_name = Path(tiff_path).name

        # If processing_context is a template string, format it
        processing_context = state.get("processing_context", "")
        try:
            state["processing_context"] = processing_context.format(
                document_name=document_name
            )
        except Exception as e:
            logger.warning(
                f"Failed to format processing_context for {document_name}: {e}"
            )
            state["processing_context"] = processing_context  # keep original

        # Clear any previous messages
        state["messages"] = []
        return state


    def chunk_and_process(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Split the image list into chunks and run the assistant on each chunk.

        Parameters
        ----------
        state : dict
            The full workflow state.  Expects:
            * state["images"] - list of image dicts (one per page)
            * state["processing_config"]["chunk_size"] - number of images per chunk
            * state["processing_context"] - user prompt for the assistant
            * state["system_prompt"] - system prompt (kept intact)

        Returns
        -------
        dict
            Updated state containing state["chunk_results"] - a list of strings.
        """
        images = state.get("images", [])
        if not images:
            logger.warning("No images found; skipping chunking.")
            state["chunk_results"] = []
            return state

        # Determine chunk size from config
        config = state.get("processing_config", {})
        chunk_size = config.get("chunk_size", 10)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            logger.warning(
                f"Invalid chunk_size ({chunk_size}); defaulting to 10."
            )
            chunk_size = 10

        total_chunks = math.ceil(len(images) / chunk_size)
        chunk_results: List[str] = []

        for idx in range(total_chunks):
            start = idx * chunk_size
            end = start + chunk_size
            chunk_imgs = images[start:end]
            message = HumanMessage(content=f"These are pages {start+1}-{end}")

            # Build a minimal state for the assistant
            chunk_state: Dict[str, Any] = {
                "system_prompt": state.get("system_prompt", "You are a helpful AI assistant."),
                "processing_config": config,          # keep detail_level, etc.
                "messages": [message],
                "image_data": chunk_imgs,             # this triggers image messages
                "user_prompt": state.get("processing_context", ""),
            }

            try:
                # Run the assistant
                out_state = self.assistant(chunk_state)
                # The assistant appends its reply as the last message
                assistant_msg = out_state["messages"][-1]
                # Safely extract content; AIMessage inherits `content`
                content = getattr(assistant_msg, "content", "")
                chunk_results.append(content)
                logger.info(f"Chunk {idx+1}/{total_chunks} processed.")
            except Exception as exc:
                # Log and continue with an empty string for this chunk
                logger.error(
                    f"Error processing chunk {idx+1}/{total_chunks}: {exc}"
                )
                chunk_results.append("")

        # Persist the results back to the workflow state
        state["chunk_results"] = chunk_results
        return state

    def consolidate_markdown(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Consolidate all chunk Markdown snippets into one final document.

        Parameters
        ----------
        state : dict
            Expects:
            * state["chunk_results"] - list of strings produced by the assistant
            * state["consolidation_context"] - a message to prefix the assistant
            * state["system_prompt"] - the system prompt (kept intact)

        Returns
        -------
        dict
            Updated state containing state["markdown_result"] - the final Markdown.
        """
        chunk_results: List[str] = state.get("chunk_results", [])
        if not chunk_results:
            logger.warning("No chunk results to consolidate.")
            state["markdown_result"] = ""
            return state

        # Join the chunk texts into a single user prompt
        user_prompt = "\n---\n".join(chunk_results)

        # Build the minimal state for the assistant
        consolidate_state: Dict[str, Any] = {
            "system_prompt": state.get("system_prompt", "You are a helpful AI assistant."),
            # Pre‑pend the consolidation context as a *Human* message
            "messages": [HumanMessage(content=state.get("consolidation_context", ""))],
            "user_prompt": user_prompt,
            # No image data - this node only handles text
            "image_data": None,
        }

        try:
            out_state = self.assistant(consolidate_state)
            assistant_msg = out_state["messages"][-1]
            content = getattr(assistant_msg, "content", "")
            state["markdown_result"] = content
            logger.info("Markdown consolidation completed.")
        except Exception as exc:
            logger.error(f"Error during markdown consolidation: {exc}")
            state["markdown_result"] = ""

        return state
