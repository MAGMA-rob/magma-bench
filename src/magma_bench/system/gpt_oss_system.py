from typing import Dict, List
import json, os, requests, asyncio

from .agents import OllamaAgent, GPTOllamaAgent, DockerAgent, map_pipeline_role_to_chat_role
from .base_system import LocalSystem


class GPTMagma(LocalSystem):
    """
    This class allows to use gpt oss with an external memory following our method.
    
    Model pretrain, no fine-tuning required.
    """

    agent : OllamaAgent

    system_prompt : str

    def __init__(self, agent_url: str, backend_url: str, backend_header : Dict, prediction_mode: str = "tool_select") -> None:
        self.commander = GPTOllamaAgent("gpt-oss:20b", backend_url, backend_header, default_params={'max_tokens':1000})

        self.base_url = agent_url
        self.max_history_length = 20

        #connection check
        try:
            response = requests.post(f"{self.base_url}/get_infos",json={},timeout=10)
            info = response.json()
            m_name = info['memorizer']
        except:
            raise ValueError(f"Connection to the Magma Agent container impossible. Please verify the url : {self.base_url}")

        self.memorizer = DockerAgent(f"{self.base_url}/update_memory", m_name)

        super().__init__({"commander":self.commander, "memorizer":self.memorizer}, prediction_mode, system_name=m_name+"+gpt_oss")
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
        query_content = self.stringify_content(query.get("content", ""))
        prompt_user = f"\nTask Attributes : {task_attributes}.\n{mem_str}\nQuery : {query_content}"

        mess = [{'role': 'system', 'content': self.system_prompt}]

        for m in self.message_history:
            mess.append({
                "role" : map_pipeline_role_to_chat_role(m.get("author")),
                "content" : m.get("content", m.get("sentence", ""))
            })

        mess.append({'role': map_pipeline_role_to_chat_role(query.get("author")), 'content': prompt_user})

        data = {
            'messages': mess
        }
        response_dict = self.commander.compute_answer(data)
        # be sure to return only one tool
        if isinstance(response_dict['action'],List):
            response_dict['action'] = response_dict['action'][0]

        say = response_dict['say']
        think = response_dict['intent']

        self._add_mess_to_history(query, say, response_dict.get("action"))

        self.compute_memory(think, say)
        
        return response_dict

    def reset_step(self):
        super().reset_step()
        self.message_history = []


BASE_SYSTEM_PROMPT = """You are a COMMANDER agent controlling a robot through optional tool calls.
You operate in a long-horizon task, but you do NOT manage persistent memory yourself.

A separate agent (the Memorizer) will handle memory updates.
Your role is to make the task state EXPLICIT so another agent can maintain memory correctly.

You will receive three structured inputs:
1. "memory": a persistent task memory summarizing past decisions, constraints, and commitments. Treat it as authoritative.
2. "task_attributes": structured metadata describing the current task state.
3. "query": the current user request or system update.

You must interpret the query using both memory and task_attributes.

Your output MUST be a JSON object with EXACTLY these three top-level fields:

{
  "intent": string,
  "say": string,
  "action": {}
}

Field semantics (strict):

- "intent":
  A short, explicit natural-language description of the task state and decision.
  This field is NOT for the user.
  It is written for another agent that will manage memory.

  The intent MUST clearly state:
  - what the user wants,
  - what constraints apply,
  - what subgoal is being executed now,
  - AND what information should be remembered, updated, or forgotten for future turns.

  When relevant, use explicit phrases such as:
  - "I need to remember that ..."
  - "The system should remember that ..."
  - "I should forget that ..."
  - "The assignment for X has changed to Y ..."
  - "This constraint remains active for future steps ..."

  The intent must be understandable without any additional context.

- "say":
  What you say to the user now.
  If an action is taken, briefly explain what you are doing.
  If no action is required, directly answer the query.

- "action":
  Use {} if no tool is required.
  Otherwise:
  {
    "name": "<tool_name>",
    "arguments": { ... }
  }

Rules (mandatory):
1. Always output valid JSON only. No text outside the JSON.
2. Never omit any of the three top-level fields.
3. Never call more than one tool.
4. Only call a tool when it is clearly required by the task logic.
5. The "intent" field must always be present and explicit.
6. Do NOT manage or modify memory directly.
7. If memory contradicts the query, memory overrides the query.
8. Do NOT include chain-of-thought or hidden reasoning.

Available tools:
"""
