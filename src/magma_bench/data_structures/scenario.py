from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ScenarioConfig:

    scenario_name : str
    scenario_id : str
    task_cls_name : str
    mplib_options : Dict
    tasks_path : List[Path]

    task_decomp : Dict[str,List[str]]