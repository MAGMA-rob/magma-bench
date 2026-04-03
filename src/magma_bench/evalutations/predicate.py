from typing import Callable, Dict, List, Tuple
from itertools import product

from magma_core.base.goals import And, At, AtLeastCountAt, BaseGoal, NotAt, On, Or
from magma_core.utils.text_utils import star_extractor


GoalBuilder = Callable[[List[str], List], BaseGoal]


def get_obs_entity_names(obs: Dict) -> List[str]:
    extra = obs.get("extra", None)
    if not isinstance(extra, dict):
        raise RuntimeError("Benchmark predicate evaluation requires obs['extra'] to be a dict.")
    return list(extra.keys())


def get_obs_entity_signature(obs: Dict) -> Tuple[str, ...]:
    return tuple(sorted(get_obs_entity_names(obs)))


def _expand_entity_arg(entity_names: List[str], value) -> List[str]:
    if isinstance(value, str):
        if "*" in value:
            return star_extractor(entity_names, value)
        return [value]
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_expand_entity_arg(entity_names, item))
        return out
    raise RuntimeError(
        f"Error in predicate argument, got type {type(value)} which is supposed "
        "to be either a List[str] or a str"
    )


def _combine_all(goals: List[BaseGoal]) -> BaseGoal:
    if len(goals) == 0:
        raise ValueError("Can not combine an empty list of goals.")
    if len(goals) == 1:
        return goals[0]
    return And(goals)


def _build_in(entity_names: List[str], args: List) -> BaseGoal:
    if len(args) != 2:
        raise ValueError(f"'in' predicate expects 2 args, got {len(args)}")
    objects = _expand_entity_arg(entity_names, args[0])
    targets = _expand_entity_arg(entity_names, args[1])
    return _combine_all([At(obj, target) for obj, target in product(objects, targets)])


def _build_not_in(entity_names: List[str], args: List) -> BaseGoal:
    if len(args) != 2:
        raise ValueError(f"'not_in' predicate expects 2 args, got {len(args)}")
    objects = _expand_entity_arg(entity_names, args[0])
    targets = _expand_entity_arg(entity_names, args[1])
    return _combine_all([NotAt(obj, [target], strict=False) for obj, target in product(objects, targets)])


def _build_on(entity_names: List[str], args: List) -> BaseGoal:
    if len(args) != 2:
        raise ValueError(f"'on' predicate expects 2 args, got {len(args)}")
    top_objects = _expand_entity_arg(entity_names, args[0])
    bottom_objects = _expand_entity_arg(entity_names, args[1])
    return _combine_all([On(top, bottom) for top, bottom in product(top_objects, bottom_objects)])


def _build_on_mutual(entity_names: List[str], args: List) -> BaseGoal:
    if len(args) != 2:
        raise ValueError(f"'on_mutual' predicate expects 2 args, got {len(args)}")
    objects_1 = _expand_entity_arg(entity_names, args[0])
    objects_2 = _expand_entity_arg(entity_names, args[1])
    return _combine_all([At(obj_1, obj_2, thresh=0.05) for obj_1, obj_2 in product(objects_1, objects_2)])


def _build_count(entity_names: List[str], args: List) -> BaseGoal:
    if len(args) != 3:
        raise ValueError(f"'count' predicate expects 3 args, got {len(args)}")
    objects = _expand_entity_arg(entity_names, args[0])
    targets = _expand_entity_arg(entity_names, args[1])
    minimum = args[2]
    if not isinstance(minimum, int):
        raise TypeError(f"'count' minimum must be an int, got {type(minimum)}")
    return _combine_all([AtLeastCountAt(objects, target, minimum) for target in targets])


def compile_predicate_to_goal(predicate: Dict, entity_names: List[str]) -> BaseGoal:
    func_name = predicate.get("predicate", "")
    args = predicate.get("args", [])

    if func_name == "or":
        if not isinstance(args, list) or len(args) == 0:
            raise ValueError("'or' predicate requires a non-empty list of sub-predicates.")
        return Or([compile_predicate_to_goal(child_predicate, entity_names) for child_predicate in args])

    builder = predicate_registry.get(func_name, None)
    if builder is None:
        raise ValueError(f"unknown predicate : {func_name}")

    if not isinstance(args, list):
        raise TypeError(f"Predicate args must be a list. Got {type(args)}")

    return builder(entity_names, args)


def compile_predicates_to_goals(predicates: List[Dict], obs: Dict) -> List[BaseGoal]:
    entity_names = list(get_obs_entity_signature(obs))
    return [compile_predicate_to_goal(predicate, entity_names) for predicate in predicates]


predicate_registry: Dict[str, GoalBuilder] = {
    "in": _build_in,
    "not_in": _build_not_in,
    "on": _build_on,
    "on_mutual": _build_on_mutual,
    "count": _build_count,
}


def evaluate_env_success(obs: Dict, goals: List[BaseGoal]) -> bool:
    for goal in goals:
        try:
            if not (goal.verify(obs) == 1).all().item():
                return False
        except Exception as e:
            print(
                f"[BENCHMARK] Fail to compute predicate {goal.name} "
                f"due to {e}. Considering it as True."
            )
            continue

    return True
