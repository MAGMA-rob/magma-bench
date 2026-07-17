from typing import Dict, List, Optional

from magma_bench.artifacts import EpisodeSpec
from magma_bench.data_structures import ScenarioConfig

class EpisodeGroup():
    """
    Manage the execution of all episodes from a Scenario
    """

    groups : List[EpisodeSpec]
    cur_idx : int


    def get_episode(self) -> Optional[EpisodeSpec]:
        if self.cur_idx >= len(self.groups):
            return None
        out = self.groups[self.cur_idx]
        self.cur_idx+=1
        return out

    
def load_groups_from_config(scenario_config : ScenarioConfig) -> List[EpisodeGroup]:
    ...