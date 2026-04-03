from typing import Dict, List
import json, requests, time, re, os

from .agents import OllamaAgent, GPTOllamaAgent, QwenOllamaAgent, DockerAgent
from .base_system import System


class StandaloneAgentSystem(System):
    """
    This class allows to test a pretrain LLM (generic) heberged on Ollama to interact with the benchmark.
    
    No external Memory provided. Classic SOA method with tool selection. No parrallel tool calls.
    """

    message_history : List[Dict]

    agent : OllamaAgent

    system_prompt : str

    def __init__(self, model_name: str, backend_url: str, backend_header : Dict, prediction_mode: str = "tool_select",**args) -> None:
        if model_name == None:
            raise ValueError("You must specify the model_name to use")
        if "gpt" in model_name:
            self.agent = GPTOllamaAgent(model_name, backend_url, backend_header, default_params={'max_tokens':1000})
        else:
            self.agent = QwenOllamaAgent(model_name, backend_url, backend_header, default_params={'max_tokens':2500})

        super().__init__({"agent":self.agent}, prediction_mode, system_name=model_name)
        self.system_prompt = BASE_SYSTEM_PROMPT
        self.message_history = []

    def init_task(self, inputs: Dict):
        super().init_task(inputs)
        tools = inputs.get("tools",[])
        self.system_prompt = BASE_SYSTEM_PROMPT + json.dumps(tools) + "\n"

    def compute_answer(self, query: Dict, task_attributes: Dict) -> Dict:
        query_content = self.stringify_content(query.get("content", ""))
        prompt_user = f"\nAttributes : {task_attributes}.\nQuery : {query_content}"

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

        self._add_mess_to_history(query,response_dict)
        
        return response_dict
    
    def _add_mess_to_history(self, query: Dict, model_answer: Dict):      
        out_list = self.agent._format_for_history(query, model_answer)
        
        self.message_history.extend(out_list)

    def reset_step(self):
        super().reset_step()
        self.message_history = []

    
BASE_SYSTEM_PROMPT = """
You are an AI model capable of calling external tools.

Your output must always be a JSON object with exactly two top-level fields:

  "say": What you would say to the user.
  "action": Either an empty JSON object or a JSON object with:
      "name": the tool name
      "arguments": a JSON object containing the tool arguments

Output exemple:
{"say":"I am launching a cycle","action":{"name":"func_name","arguments":{"param1":val1,"param2":val2...}}}

Rules:
1. Never call more than one tool.
2. Only call a tool when it is clearly helpful or required. Use only existing tool.
3. If calling a tool, "say" should briefly explain what you are doing.
4. If no tool is needed, set "action": {}. "say" must directly be the answer to the user request.
5. Always output valid JSON only, with no text outside the JSON.
6. The output json must always contains a "say" and a "action".

Available tools:
"""
