from .agents import Agent

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import threading
import json

from magma_core.utils.text_utils import (
    format_history_message,
    format_model_history_message,
)

class System(ABC):
    """
    This class englobs your multi-agent systems or unique agents to interact with the Benchmark
    
    You can inherits directly from it if your model server already handles memory, history etc..
    Otherwise, use `LocalSystem` class instead.
    """

    system_name : str

    # Stock each agents used with a key corresponding to their role
    agents : Dict[str, Agent]

    # Stock the current tool API avalaible
    tools : List[Dict]

    def __init__(self, agents : Dict[str, Agent], prediction_mode : str = "sequence", **args) -> None:
        if prediction_mode != "sequence" and prediction_mode != "tool_select":
            raise ValueError(f"prediction_mode should be sequence or tool_select. Got {prediction_mode}")
        self.prediction_mode = prediction_mode
        name_seg : List[str] = args.get("system_name","default_system_name").split("/")
        self.system_name = name_seg[-1] if name_seg[-1].strip() != "" else name_seg[-2]
        self.agents = agents
        self.tools = []

    def init_task(self, inputs : Dict):
        self.tools = inputs.get("tools",[])

    @staticmethod
    def stringify_content(content: Any) -> str:
        if isinstance(content, (dict, list)):
            return json.dumps(content, ensure_ascii=True)
        if content is None:
            return ""
        return str(content)

    @abstractmethod
    def compute_answer(self, query : Dict, task_attributes : Dict) -> Dict:
        """
        Main function to compute the answer from a a query and task attributes
        
        It must return a dict with a key "action" and a key "say". Action is the tool call and say is the answer to the user. 
        "think" could also be included in the out dict for logging inside result.
        """
        raise NotImplementedError()
    
    def get_system_card(self) -> Dict:
        """Return a Dict with details on the system (used agents,..). It's used to save a card next to the result to trace which model you use."""
        
        agents = {
            agent_name : agent_ref.agent_name for agent_name, agent_ref in self.agents.items()
        }
        return {
            "agents" : agents,
            "prediction_mode" : self.prediction_mode
        }

    def reset_task(self):
        """Allows to reset system information between evaluation tasks"""
        self.tools = []
        self.reset_step()

    def reset_step(self):
        """Allows to reset system state between steps"""
        for role, agent in self.agents.items():
            agent.reset()

    def get_listed_memory(self):
        """Return the memory of the model"""
        return []

class LocalSystem(System, ABC):
    """
    This class allows you to define specific logic inside the system if you do not want to modify code inside your model server.
    
    Typically, it manages the memory and the history of messages.
    But you can add some specific format conversion etc...
    """

    # Memory
    memory : List[str]
    preserved_memory_indices : List[int] # indices corresponding to static constraints
    memory_update_lock = threading.Lock()

    # History
    message_history : List[Dict] # historic of messages
    max_history_length : int = 4 # maximal number of messages
    time_window : float = 90 # maximal time windows

    def __init__(self, agents: Dict[str, Agent], prediction_mode : str = "sequence", max_time_window : float = 90, **args) -> None:
        super().__init__(agents, prediction_mode, **args)
        self.time_window = max_time_window
        self.message_history = []
        self.memory = []
        self.preserved_memory_indices = []

    def init_task(self, inputs: Dict):
        super().init_task(inputs)
        self.memory = inputs.get("memory",[])
        self.preserved_memory_indices = inputs.get("preserved_memory_indices",[])

    def _add_mess_to_history(
        self,
        query: Dict,
        model_answer: Any,
        model_action: Any = None,
        model_answer_timestamps=None,
    ) -> None:
        """
        Append the interaction step to the message history.
        """
        if not model_answer_timestamps or not isinstance(model_answer_timestamps, float):
            model_answer_timestamps = query['timestamp']+10
        
        self.message_history.extend(
            [
                format_history_message(
                    query.get("author","user"),
                    query.get("content"),
                    query["timestamp"],
                ),
                format_model_history_message(
                    model_answer,
                    model_action,
                    model_answer_timestamps,
                ),
            ])

        while len(self.message_history) > self.max_history_length:
            # Remove the oldest message if we exceed the max history length
            self.message_history.pop(0)

    def _get_recent_messages(self, current_time: float) -> List[Dict]:
        """
        Get messages from the last time_window seconds.

        Args:
            current_time (float): The current timestamp in seconds.

        Returns:
            List[Dict]: A list of recent messages.
        """
        self.message_history = [
            msg for msg in self.message_history
            if current_time - msg["timestamp"] <= self.time_window
        ]

        return self.message_history
    
    def _get_memory(self) -> str:
        """Return a str for the memory"""
        mem = "Memory:\n"
        for i, m in enumerate(self.memory):
            if i in self.preserved_memory_indices:
                id = "X"
            else:
                id = i
            mem += f"[{id}] {m}\n"
        return mem
    
    def get_listed_memory(self):
        return self.memory.copy()
    
    def compute_memory(self, think : str, say : str):
        """
        Compute the memory update of the system.
        """
        if not "memorizer" in self.agents:
            raise ValueError("You must define a 'memorizer' agents in the __init__ to use the LocalSystem")
        
        with self.memory_update_lock:
            payload = {
                "think": think,
                "say": say,
                "memory": self.memory,
                "preserved_memory_indices" : self.preserved_memory_indices
            }

            response_dict = self.agents['memorizer'].compute_answer(payload)

            update_list = response_dict.get("update", None)
            if not update_list:
                print(f"[{self.system_name}] No update found in response: {response_dict}")
                raise Exception("No update found in response")
            
            self._update_memory(update_list)

    def _update_memory(self, update_list : List):
        for update_dict in update_list:
            if update_dict["name"] == "add":
                self.memory.extend(update_dict["arguments"]["statements"])
            elif update_dict["name"] == "remove":
                for i in update_dict["arguments"]["ids"]:
                    try:
                        id = int(i)
                        if id in self.preserved_memory_indices or id >= len(self.memory):
                            continue
                        self.memory.remove(self.memory[id])
                    except:
                        continue
            else:
                print(f"[{self.system_name}] Unknown update operation: {update_dict['name']}")
                raise Exception(f"Unknown update operation: {update_dict['name']}")

    def reset_step(self):
        self.memory = []
        self.message_history = []
        return super().reset_step()
