from abc import abstractmethod, ABC
from typing import Dict, List

class Agent(ABC):
    """
    Base class to define an agent LLM to interact inside our system
    """

    url : str
    agent_name : str

    def __init__(self, url : str, agent_name : str) -> None:
        self.url = url
        self.agent_name = agent_name

    @abstractmethod
    def compute_answer(self, inputs : Dict) -> Dict:
        """Main function to compute the answer from a dict of inputs"""
        raise NotImplementedError()
    
    @abstractmethod
    def reset(self):
        """Allows to reset the state of the agent betweens benchamrk steps"""
        pass

    def get_infos(self) -> Dict:
        return {
            "name" : self.agent_name
        }