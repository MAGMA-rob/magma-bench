from typing import Dict
from .base_agent import Agent

import time, requests
from requests.exceptions import ConnectionError, RequestException

class DockerAgent(Agent):
    """This class allows to define a custom logic for an agent which runs in a separate docker or any code accessible through an url"""

    def compute_answer(self, inputs: Dict) -> Dict:
        inputs.setdefault("inference_mode",True)
        try:
            ti = time.time()
            response = requests.post(self.url, json=inputs, timeout=120)
            ti = time.time() - ti
            response.raise_for_status()
        except ConnectionError as e:
            # e.g., ConnectionRefusedError, DNS failure, etc.
            raise ConnectionError(f"[MagmaLLM] Connection error: {e}")
        except RequestException as e:
            raise RequestException(f"[MagmaLLM] Request failed: {e}\n Message was {inputs}")
        return response.json()
    
    def reset(self):
        return super().reset()