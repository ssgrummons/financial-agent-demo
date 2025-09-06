from abc import ABC, abstractmethod
from langchain_core.tools import BaseTool, tool
from typing import List

class CustomToolSetBase(ABC):
    """
    Generic class for custom toolsets
    """
    @abstractmethod
    def load_tools(self) -> List[BaseTool]:
        """
        Abstract method for loading tools
        """
        pass

class FinanceTools(CustomToolSetBase):
    """
    Class containing the tools required for Finance
    """
    def __init__(self):
        """
        Initialize the FinanceTools
        """
        pass
        
    def load_tools(self) -> List[BaseTool]:
        """
        Load and return finance-related tools
        """
        # Return empty list for now - you'll build this out later
        return []