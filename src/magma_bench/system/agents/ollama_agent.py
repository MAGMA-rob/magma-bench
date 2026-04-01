from typing import Dict, List, Optional
from abc import ABC, abstractmethod

import time, requests, json, re
from .base_agent import Agent


def _stringify_message_content(content) -> str:
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=True)
    if content is None:
        return ""
    return str(content)


class OllamaAgent(Agent, ABC):
    """This class allows to use an agent on Ollama to interact with the benchmark"""

    model_name : str
    default_params : Dict
    backend_header : Dict

    def __init__(self, model_name : str, backend_url: str, backend_header : Dict, default_params : Dict = {}) -> None:
        self.model_name = model_name
        self.default_params = default_params
        self.backend_header = backend_header
        super().__init__(backend_url, model_name)

    def compute_answer(self, inputs: Dict) -> Dict:
        inputs.setdefault("model", self.model_name)
        data = self.default_params.copy()
        data.update(inputs)
        try:
            response = requests.post(self.url, headers=self.backend_header, data=json.dumps(data))
            response_dic = self._extract_model_answer(response)        
        except Exception as e:
            print(e)
            response_dic = {
                "reasoning": "",
                "action": {},
                "say": "-"
            }
        return response_dic
    
    def _content_to_json(self, content : str) -> Optional[Dict]:
        try:
            return json.loads(content)
        except:
            return None
    
    @abstractmethod
    def _extract_model_answer(self, response) -> Dict:
        raise NotImplementedError()
    
    @abstractmethod
    def _format_for_history(self, query : Dict, model_answer: Dict) -> List:
        raise NotImplementedError
    
    def reset(self):
        return super().reset()
    
class GPTOllamaAgent(OllamaAgent):
    """An agent to manage chat with model which have proper reasonning and content file separated."""

    def _extract_model_answer(self, response) -> Dict:
        r = response.json()['choices'][0]['message']
        
        # sometimes gpt use tool_calls
        if 'tool_calls' in r:
            print("Alternative path ")
            print(r)

            def _convert(t) -> Dict:
                if isinstance(t, str):
                    t = json.loads(t)
                if isinstance(t["arguments"], str): 
                    t["arguments"] = json.loads(t["arguments"])
                return t

                
            if isinstance(r['tool_calls'],str): r['tool_calls'] = json.loads(r['tool_calls'])
            if isinstance(r['tool_calls'], List):
                ac = []
                for tool_call in r['tool_calls']:
                    ac.append(_convert(tool_call['function']))
            else:
                ac = _convert(r['tool_calls']['function'])

            return {
                "action" : ac,
                "reasoning" : r['reasoning'],
                "say" : r['content']
            }

        
        response_dic = self._content_to_json(r['content'])
        if not response_dic or not "say" in response_dic or not "action" in response_dic:
            print("BAD FORMAT DETECTED : ", r)
            return {
                "intent" : r['reasoning'],
                "say" : r['content'],
                "action" : {}
            }

        response_dic["reasoning"] = r["reasoning"]

        return response_dic
    
    def _format_for_history(self, query : Dict, model_answer: Dict) -> List:
        return [
            {"role":query.get("author"), "content":_stringify_message_content(query.get('content'))},
            {"role":"assistant","content":str(model_answer['say']) + "\n" + str(model_answer['action']),"reasoning":model_answer['reasoning']}
        ]
    
class ManagedGPTOllamaAgent(OllamaAgent):
    """An agent to manage chat with model which have proper reasonning and content file separated."""

    def _extract_model_answer(self, response) -> Dict:
        r = response.json()['choices'][0]['message']
        
        # sometimes gpt use tool_calls
        if 'tool_calls' in r:
            print("PASSING IN ALTERNATIVE")
            def _convert(t) -> Dict:
                if isinstance(t, str):
                    t = json.loads(t)
                if isinstance(t["arguments"], str): 
                    t["arguments"] = json.loads(t["arguments"])
                return t

                
            if isinstance(r['tool_calls'],str): r['tool_calls'] = json.loads(r['tool_calls'])
            if isinstance(r['tool_calls'], List):
                ac = []
                for tool_call in r['tool_calls']:
                    ac.append(_convert(tool_call['function']))
            else:
                ac = _convert(r['tool_calls']['function'])

            return {
                "action" : ac,
                "reasoning" : r['reasoning'],
                "say" : r['content'],
                "memory_update": {'add': [], 'remove':[]},
            }

        
        response_dic = self._content_to_json(r['content'])
        if not response_dic or not "say" in response_dic or not "action" in response_dic:
            return {
                "reasoning" : r['reasoning'],
                "say" : r['content'],
                "memory_update": {'add': [], 'remove':[]},
                "action" : {}
            }

        response_dic["reasoning"] = r["reasoning"]

        return response_dic
    
    def _format_for_history(self, query : Dict, model_answer: Dict) -> List:
        return [
            {"role":query.get("author"), "content":_stringify_message_content(query.get('content'))},
            {"role":"assistant","content":str(model_answer['say']) + "\n" + str(model_answer['action']),"reasoning":model_answer['reasoning']}
        ]
    
class QwenOllamaAgent(OllamaAgent):
    """An agent to manage chat with model which use <think> to delimitate the think process."""

    def _extract_model_answer(self, response) -> Dict:
        r = response.json()['choices'][0]['message']   
        match = re.search(r"<think>(.*?)</think>\s*(.*)", r['content'], re.DOTALL)
        if match:
            think_text = match.group(1).strip()
            rest_text = match.group(2).strip()
        else:
            # Fallback: detect only </think>
            end_match = re.search(r"(.*?)</think>\s*(.*)", r['content'], re.DOTALL)
            if end_match:
                think_text = end_match.group(1).strip()
                rest_text = end_match.group(2).strip()
            else:
                think_text = ""
                rest_text = r['content']
        
        response_dic = self._content_to_json(rest_text)
        if response_dic:
            ac = response_dic.get("action",None)
            say = response_dic.get("say",None)
            if ac != None and say != None:
                return {
                    "action" : response_dic['action'],
                    "reasoning" : think_text,
                    "say" : response_dic['say']
                }
        
        return {
            "action" : {},
            "reasoning" : think_text,
            "say" : rest_text
        }
    
    def _format_for_history(self, query: Dict, model_answer: Dict) -> List:
        return [
            {"role":query.get("author"), "content":_stringify_message_content(query.get('content'))},
            {"role":"assistant","content":str(model_answer['say'])}
        ]

class ManagedQwenOllamaAgent(QwenOllamaAgent):
    """An agent to manage chat with model which use <think> to delimitate the think process."""

    def _extract_model_answer(self, response) -> Dict:
        r = response.json()['choices'][0]['message']   
        match = re.search(r"<think>(.*?)</think>\s*(.*)", r['content'], re.DOTALL)
        if match:
            think_text = match.group(1).strip()
            rest_text = match.group(2).strip()
        else:
            # Fallback: detect only </think>
            end_match = re.search(r"(.*?)</think>\s*(.*)", r['content'], re.DOTALL)
            if end_match:
                think_text = end_match.group(1).strip()
                rest_text = end_match.group(2).strip()
            else:
                think_text = ""
                rest_text = r['content']
        
        response_dic = self._content_to_json(rest_text)
        if response_dic:
            ac = response_dic.get("action",None)
            say = response_dic.get("say",None)
            if ac != None and say != None:
                return {
                    "memory_update": response_dic.get("memory_update",{'add': [], 'remove':[]}),
                    "action" : response_dic['action'],
                    "reasoning" : think_text,
                    "say" : response_dic['say']
                }
        
        return {
            "memory_update" : {'add': [], 'remove':[]},
            "action" : {},
            "reasoning" : think_text,
            "say" : rest_text
        }
