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
    Class containing the tools required for Coding
    """
    def load_tools(self) -> List[BaseTool]:
        """
        """