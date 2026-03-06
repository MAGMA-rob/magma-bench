from typing import Dict, List
import json, os, requests, asyncio

from .agents import OllamaAgent, QwenOllamaAgent, DockerAgent
from .base_system import LocalSystem

class QwenMagma(LocalSystem):
    """
    This class allows to use qwen style with an external memory following our method.
    
    Model pretrain, no fine-tuning required.
    """

    agent : OllamaAgent

    system_prompt : str

    def __init__(self, agent_url : str, backend_url: str, backend_header : Dict, model_name : str, prediction_mode: str = "tool_select") -> None:
        if model_name == None:
            raise ValueError("You must specify the ollama instance id to use.")
        self.commander = QwenOllamaAgent(model_name, backend_url, backend_header, default_params={'max_tokens':2500})
        self.base_url = agent_url

        #connection check
        try:
            response = requests.post(f"{self.base_url}/get_infos",json={},timeout=10)
            info = response.json()
            m_name = info['memorizer']
        except:
            raise ValueError(f"Connection to the Magma Agent container impossible. Please verify the url : {self.base_url}")

        self.memorizer = DockerAgent(f"{self.base_url}/update_memory", m_name)

        super().__init__({"commander":self.commander, "memorizer":self.memorizer}, prediction_mode, system_name=model_name+"+"+m_name)
        self.system_prompt = BASE_SYSTEM_PROMPT

    def init_task(self, inputs: Dict):
        super().init_task(inputs)
        tools = inputs.get("tools",[])
        self.system_prompt = BASE_SYSTEM_PROMPT + json.dumps(tools) + "\n"

    def compute_answer(self, query: Dict, task_attributes: Dict) -> Dict:
        mem_str = "Memory:\n"
        if len(self.memory) > 0:
            for mem in self.memory:
                mem_str += f"- {mem}\n"
        else:
            mem_str += "empty\n"
        prompt_user = f"\nTask Attributes : {task_attributes}.\n{mem_str}\nQuery : {query.get('content','')}"

        mess = [{'role': 'system', 'content': self.system_prompt}]

        for m in self.message_history:
            mess.append({
                "role" : m['author'],
                "content" : m['sentence']
            })

        mess.append({'role': 'user', 'content': prompt_user})

        data = {
            'messages': mess
        }
        
        response_dict = self.commander.compute_answer(data)

        # be sure to return only one tool
        if isinstance(response_dict['action'],List):
            response_dict['action'] = response_dict['action'][0]

        say = response_dict['say']
        think = response_dict['reasoning']

        self._add_mess_to_history(query,f"{say}, {response_dict['action']}")

        asyncio.run(self.compute_memory(think, say))
        
        return response_dict

    def reset_step(self):
        super().reset_step()
        self.message_history = []


BASE_SYSTEM_PROMPT = """
You are an AI model capable of calling external tools to control a robot.

You will receive a user message containing three structured elements:
1. "memory": a compressed representation of past context. Treat it as authoritative and mandatory to follow. It provides constraints, intentions, past states, and commitments.
2. "task_attributes": a dictionary describing the current task state, parameters, or metadata. These attributes must guide and constrain your decisions.
3. "query": the current request, which may come from the user or from an internal system/tool status update.

Your job is to:
- Interpret the "query" in the context of both "memory" and "task_attributes".
- Use "memory" to preserve consistency across long-running tasks.
- Use tools only when required by the task logic and constraints.
- Produce a structured JSON output describing what to say and what action to perform.

Your output must always be a JSON object with exactly two top-level fields:

  "say": What you would say to the user.
  "action": Either an empty JSON object {} or a JSON object with:
      "name": the tool name
      "arguments": a JSON object containing the tool arguments

Output example:
{"say": "I am launching a cycle", "action": {"name": "func_name", "arguments": {"param1": val1, "param2": val2}}}

Rules:
1. Never call more than one tool.
2. Only call a tool when it is clearly helpful or required. Use only existing tools.
3. "memory" must always be respected. It overrides assumptions and should guide interpretation of the user's intent.
4. "task_attributes" must be integrated into your reasoning. If they contradict the query, resolve the contradiction logically using "memory" first, then the attributes.
5. If calling a tool, "say" should briefly explain what you are doing.
6. If no tool is needed, set "action": {} and "say" must directly answer the query.
7. Always output valid JSON only, with no text outside the JSON.
8. The output JSON must always contain both "say" and "action".
9. Never ignore or alter the structure of the input (memory, task_attributes, query). Use them explicitly in your reasoning and final decision.

Available tools:
"""