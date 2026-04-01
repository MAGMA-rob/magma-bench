from typing import Dict, List
import time, os, json
import asyncio, threading, requests

from .base_system import LocalSystem
from .agents import Agent, DockerAgent

class MagmaSingle(LocalSystem):
    """
    Class containing the interaction with class Magma system single-agent.
    It communicates with a Magma agent server.
    No memory management. The full interaction history is concatenated for each step.
    """

    # Different Agent used with the Magma System
    commander : Agent

    def __init__(self, agent_url : str, **args) -> None:
        """
        Initialize the MagmaSystem agent.
        """
        self.base_url = agent_url

        # Connection check
        try:
            response = requests.post(f"{self.base_url}/get_infos",json={},timeout=10)
            info = response.json()
            print(info)
            c_name = info['commander']
        except:
            raise ValueError(f"Connection to the Magma Agent container impossible. Please verify the url : {self.base_url}")

        self.commander = DockerAgent(f"{self.base_url}/chat", c_name)

        args['system_name'] = c_name

        super().__init__({"commander":self.commander}, prediction_mode="tool_select", **args)
        

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
        memory_update = response_dict.get("memory_update", None)
        if ac == None or say == None or think == None:
            print(f"[MagmaLLM] Error in Format: {response_dict}")
            return {"error": "No action found in response"}

        self._add_mess_to_history(query, say, t + model_mess_time)

        if isinstance(ac, str):
            try:
                response_dict['action'] = json.loads(ac)
            except:
                response_dict["action"] = {}

        if memory_update:
            self._update_memory([
            {"name":"add","arguments":{"statements":memory_update.get('add',[])}},
            {"name":"remove","arguments":{"ids":memory_update.get('remove',[])}}
            ])
            print(memory_update)
            

        return response_dict
    
    def _get_recent_messages(self, current_time: float) -> List[Dict]:
        return self.message_history
    
    def _add_mess_to_history(self, query: Dict, model_answer: str, model_answer_timestamps=None) -> None:
        if not model_answer_timestamps or not isinstance(model_answer_timestamps, float):
            model_answer_timestamps = query['timestamp']+10
        
        self.message_history.extend(
            [
                {
                    "author": query['author'],
                    "sentence": self.stringify_content(query['content']),
                    "timestamp": query['timestamp']
                },
                {
                    "author" : "model",
                    "sentence": model_answer,
                    "timestamp": model_answer_timestamps
                }
            ])
