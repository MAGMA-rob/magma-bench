from typing import Dict, List
import json, requests, time, re, os

from .agents import OllamaAgent, ManagedGPTOllamaAgent, ManagedQwenOllamaAgent, DockerAgent
from .base_system import LocalSystem

class SelfManagedAgentSystem(LocalSystem):
    """
    This class allows to test a pretrain LLM (generic) heberged on Ollama to interact with the benchmark.
    
    No external Memory provided. The memory is self-managed toward the direct model execution.
    Classic SOA method with tool selection. No parrallel tool calls.
    """

    message_history : List[Dict]

    agent : OllamaAgent

    system_prompt : str

    def __init__(self, model_name: str, backend_url: str, backend_header : Dict, prediction_mode: str = "tool_select",**args) -> None:
        if model_name == None:
            raise ValueError("You must specify the model_name to use")
        if "gpt" in model_name:
            self.agent = ManagedGPTOllamaAgent(model_name, backend_url, backend_header, default_params={'max_tokens':1000})
        else:
            self.agent = ManagedQwenOllamaAgent(model_name, backend_url, backend_header, default_params={'max_tokens':2500})

        super().__init__({"agent":self.agent}, prediction_mode, system_name=f"SM_{model_name}")
        self.system_prompt = BASE_SYSTEM_PROMPT
        self.message_history = []

    def init_task(self, inputs: Dict):
        super().init_task(inputs)
        tools = inputs.get("tools",[])
        self.system_prompt = BASE_SYSTEM_PROMPT + json.dumps(tools) + "\n"

    def compute_answer(self, query: Dict, task_attributes: Dict) -> Dict:
        query_content = self.stringify_content(query.get("content", ""))
        prompt_user = f"\nMemory:\n{self._get_memory()}\nTask Attributes : {task_attributes}.\nQuery : {query_content}"

        mess = [{'role': 'system', 'content': self.system_prompt}]

        for m in self.message_history:
            mess.append(m)

        mess.append({'role': 'user', 'content': prompt_user})

        data = {
            'messages': mess
        }
        response_dict = self.agent.compute_answer(data)
        # be sure to return only one tool
        if isinstance(response_dict['action'],List):
            response_dict['action'] = response_dict['action'][0]

        out_list = self.agent._format_for_history(query, response_dict)
        self.message_history.extend(out_list)

        self._update_memory([
            {"name":"add","arguments":{"statements":response_dict['memory_update']['add']}},
            {"name":"remove","arguments":{"ids":response_dict['memory_update']['remove']}}
        ])
        
        return response_dict
    
    def _add_mess_to_history(self, query: Dict, model_answer: str, model_answer_timestamps=None) -> None:
        raise NotImplementedError


BASE_SYSTEM_PROMPT = """You are a COMMANDER agent controlling a robot through optional tool calls.
You operate in a long-horizon task using an external memory.

You will receive three structured inputs:
1. "memory": 
    - an authoritative, persistent task memory.
    - It contains constraints, commitments, goals, and important past facts.
    - You are able to add element to it using a memory_update field in your ouput.
    - You can also remove element from it by adding the id to the memory_update_field.
2. "task_attributes":
    - structured metadata about the current task state.
3. "query":
    - the current user request or system feedback.

You MUST reason using all three inputs.
Your output MUST be a JSON object with EXACTLY these three top-level fields:

{
  "say": string,
  "action": {}
  "memory_update": {
    "add": string[],
    "remove": int[]
  },
}

Field semantics (strict):
- "say":
  What you say to the user now. If you take an action, briefly explain it.
  If no action is needed, directly answer the query.

- "memory_update":
  - "add": list(str) Facts, commitments, or constraints that must persist across future turns.
  - "remove": list(int) Ids of element that you want to remove from the memory.
  If no memory change is required, both lists must be empty.

- "action":
  Use {} if no tool is required.
  Otherwise:
  {
    "name": "<tool_name>",
    "arguments": { ... }
  }

Rules (mandatory):
1. Always output valid JSON only. No text outside the JSON.
2. Never omit any of the four top-level fields.
3. Never call more than one tool.
4. Only call a tool when it is clearly required by the task logic.
5. Memory updates must be explicit. Never rely on implicit memory.
6. If "memory" contradicts the "query", the memory overrides the query.
7. Do not include chain-of-thought or hidden reasoning.

Available tools:
"""
