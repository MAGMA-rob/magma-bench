from typing import List, Dict, Callable, Any
import torch
from itertools import product

from magma_core.utils.text_utils import star_extractor

def _in(env_state_dict : Dict, object : str, target : str) -> bool:
    dist = torch.norm(env_state_dict[object][0][:2] - env_state_dict[target][0][:2])
    return (dist < 0.1).item()

def _on(env_state_dict, top_object, bot_object) -> bool:
    dist = torch.norm(env_state_dict[top_object][0][:2] - env_state_dict[bot_object][0][:2])
    return (dist < 0.05 and env_state_dict[top_object][0][2] > env_state_dict[bot_object][0][2]).item()

def _on_mutual(env_state_dict, object1, object2) -> bool:
    dist = torch.norm(env_state_dict[object1][0][:2] - env_state_dict[object2][0][:2])
    return (dist < 0.05).item()

def _not_in(env_state_dict, object, target) -> bool:
    return not _in(env_state_dict, object, target)

def _or(env_state_dict, *predicates) -> bool:
    for predicate in predicates:
        try:
            out = eval_predicate(env_state_dict, predicate)
        except Exception as e:
            print(f"[BENCHMARK] Fail to compute predicate {predicate.get('predicate','UNKNOWN')} due to {e}. Considering it False")
            continue
        if out:
            return out
    return False

def lift_predicate(func: Callable) -> Callable:
    def wrapper(env_state_dict, *args):
        normalized = [
            arg if isinstance(arg, list) else [arg]
            for arg in args
        ]

        # Cartesian product of all arguments
        for combo in product(*normalized):
            if not func(env_state_dict, *combo):
                return False
        return True

    return wrapper

predicate_know = {
    "in" : lift_predicate(_in),
    "not_in" : lift_predicate(_not_in),
    "on" : lift_predicate(_on),
    "on_mutual" : lift_predicate(_on_mutual),
    "or" : lift_predicate(_or)
}

###############################

def recursive_extractor(env_actors, args) -> List:
    out_args = []
    for arg in args:
        if isinstance(arg, List):
            out_args.append(recursive_extractor(env_actors, arg))
        elif isinstance(arg, str):
            if '*' in arg:
                out_args.append(star_extractor(env_actors, arg))
            else:
                out_args.append(arg)
        else:
            raise RuntimeError(f"Error in predicate argument, got type {type(arg)} which is supposed to be either a List[str] or a str")
    return out_args

def eval_predicate(env_state_dict : Dict, predicate : Dict):
    func_name = predicate.get("predicate","")
    args = predicate.get("args", [])

    func = predicate_know.get(func_name, None)

    if not func:
        raise ValueError(f"unknown predicate : {func_name}")

    args = recursive_extractor(list(env_state_dict.keys()),args)

    return func(env_state_dict, *args)

def evaluate_env_success(obs, output_predicates : List[Dict]) -> bool: #and
    out = True
    for predicate in output_predicates:
        try:
            out = eval_predicate(obs, predicate)
        except Exception as e:
            print(f"[BENCHMARK] Fail to compute predicate {predicate.get('predicate','UNKNOWN')} due to {e}. Considering it as True.")
            continue

        if not out:
            break
    return out
