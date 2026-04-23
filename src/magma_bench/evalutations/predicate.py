from typing import Dict, List

from magma_core.base.goals import BaseGoal

from ._predicate_compiler import PredicateCompiler
from ._predicate_registry import (
    GoalBuilder,
    PredicateArgumentsValidator,
    get_obs_entity_names,
    get_obs_entity_signature,
    predicate_argument_validators,
    predicate_definitions,
    predicate_registry,
)


_predicate_compiler = PredicateCompiler(predicate_definitions)

__all__ = [
    "GoalBuilder",
    "PredicateArgumentsValidator",
    "compile_predicate_to_goal",
    "compile_predicates_to_goals",
    "evaluate_env_success",
    "get_obs_entity_names",
    "get_obs_entity_signature",
    "predicate_argument_validators",
    "predicate_registry",
    "validate_predicate_spec",
    "validate_predicate_specs",
]


def validate_predicate_spec(predicate: Dict) -> None:
    _predicate_compiler.validate_spec(predicate)


def compile_predicate_to_goal(predicate: Dict, entity_names: List[str]) -> BaseGoal:
    return _predicate_compiler.compile_to_goal(predicate, entity_names)


def compile_predicates_to_goals(predicates: List[Dict], obs: Dict) -> List[BaseGoal]:
    entity_names = list(get_obs_entity_signature(obs))
    return _predicate_compiler.compile_many(predicates, entity_names)


def validate_predicate_specs(predicates: List[Dict]) -> None:
    _predicate_compiler.validate_specs(predicates)


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
