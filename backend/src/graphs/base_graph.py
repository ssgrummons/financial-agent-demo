from abc import ABC, abstractmethod
from langgraph.graph import StateGraph
from typing import Type, TypeVar, Generic
from components.shared_nodes import SharedNodes
from pydantic import BaseModel
from langgraph.checkpoint.memory import InMemorySaver

S = TypeVar("S", bound=BaseModel)

class BaseDocumentGraph(Generic[S], ABC):
    """
    Base class for all document‑processing graphs.

    Parameters
    ----------
    state_schema : Type[S]
        Pydantic model that describes the shape of the state that will
        flow through the graph.
    """
    def __init__(self, state_schema: Type[S]):
        self.state_schema = state_schema
        # Shared helper nodes
        self.shared_nodes = SharedNodes()
        # The graph instance will be created by the subclass
        self.graph: StateGraph[S] = self.build_graph()

    @abstractmethod
    def build_graph(self) -> StateGraph[S]:
        """
        Sub‑classes must implement this to construct the graph.

        Returns
        -------
        StateGraph[S]
            A fully wired graph that can be compiled.
        """
        pass

    def create_compiled_graph(self) -> StateGraph[S]:
        """
        Compile the graph after it has been built.
        """
        memory = InMemorySaver()
        return self.graph.compile(checkpointer=memory)

    def add_common_nodes(self, graph: StateGraph[S]) -> StateGraph[S]:
        """
        Attach nodes that every graph should expose.

        Parameters
        ----------
        graph : StateGraph[S]
            The graph instance to which we add the common nodes.

        Returns
        -------
        StateGraph[S]
            The graph instance with the common nodes added.
        """
        graph.add_node("validate_input", self.shared_nodes.validate_input)
        graph.add_node("load_document_content", self.shared_nodes.load_document_content)
        graph.add_node("log_progress", self.shared_nodes.log_progress)
        return graph