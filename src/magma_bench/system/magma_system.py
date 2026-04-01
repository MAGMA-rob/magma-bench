from typing import Dict, List
import time, os, json
import requests

from .base_system import LocalSystem
from .agents import Agent, DockerAgent

class MagmaDual(LocalSystem):
    """
    Class containing the interaction with class Magma system dual-agent.
    """

    # Different Agent used with the Magma System
    commander : Agent
    memorizer : Agent

    def __init__(self, agent_url, **args) -> None:
        """
        Initialize the MagmaDual system.
        It connects to a Magma agent server that contains two agents: a commander and a memorizer.
        """
        self.base_url = agent_url

        # Connection check
        try:
            response = requests.post(f"{self.base_url}/get_infos",json={},timeout=10)
            info = response.json()
            c_name = info['commander']
            m_name = info['memorizer']
        except:
            raise ValueError(f"Connection to the Magma Agent container impossible. Please verify the url : {self.base_url}")

        self.commander = DockerAgent(f"{self.base_url}/chat", c_name)
        self.memorizer = DockerAgent(f"{self.base_url}/update_memory", m_name)

        args['system_name'] = f"{c_name}+{m_name}"

        super().__init__({"commander":self.commander,"memorizer":self.memorizer}, prediction_mode="tool_select", **args)
        

    def compute_answer(self, query: Dict, task_attributes: Dict) -> Dict:
        if not task_attributes or not query:
            raise ValueError("Inputs must contain 'perception' and 'query' fields")
        
        t = query.get("timestamp", 0)

        history = self._get_recent_messages(t)
        
        with self.memory_update_lock:
            payload = {
                "instruction": self.stringify_content(query.get("content", "")),
                "attributes": task_attributes,
                "memory": self.memory,
                "function": self.tools,
                "history": history
            }

        model_start_time = time.time()
        response_dict = self.commander.compute_answer(payload)
        model_mess_time = time.time() - model_start_time

        ac = response_dict.get("action", None)
        say = response_dict.get("say", None)
        think = response_dict.get("think", None)
        if ac == None or say == None or think == None:
            print(f"[MagmaLLM] Error in Format: {response_dict}")
            return {"error": "No action found in response"}

        self._add_mess_to_history(query, say, t + model_mess_time)

        if isinstance(ac, str):
            try:
                response_dict['action'] = json.loads(ac)
            except:
                response_dict["action"] = {}
        self.compute_memory(think, say)
        return response_dict
