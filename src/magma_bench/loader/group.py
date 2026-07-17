from typing import List, Optional

from magma_bench.data_structures import Episode, Scenario

class EpisodeGroup():
    """
    Manage the execution of all episodes from a Scenario
    """

    groups : List[Episode]
    cur_idx : int


    def get_episode(self) -> Optional[Episode]:
        if self.cur_idx >= len(self.groups):
            return None
        out = self.groups[self.cur_idx]
        self.cur_idx+=1
        return out

    
def load_groups_from_config(scenario: Scenario) -> List[EpisodeGroup]:
    ...
