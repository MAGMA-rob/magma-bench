import inspect
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, List, Tuple

from magma_core.base.goals import And, At, AtLeastCountAt, BaseGoal, NotAt, On
from magma_core.utils.text_utils import star_extractor


GoalBuilder = Callable[[List[str], Dict[str, Any]], BaseGoal]
PredicateArgumentsValidator = Callable[[Any], Dict[str, Any]]


@dataclass(frozen=True)
class PredicateDefinition:
    name: str
    validator: PredicateArgumentsValidator
    builder: GoalBuilder


def validate_entity_selector(value: Any, context: str) -> None:
    if isinstance(value, str):
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            validate_entity_selector(item, f"{context}[{idx}]")
        return
    raise TypeError(
        f"{context} must be either a str or a nested list of str. Got {type(value)}."
    )


def validate_threshold(value: Any, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a float. Got {type(value)}.")


def validate_string(value: Any, context: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a str. Got {type(value)}.")


def validate_predicate_arguments(predicate_name: str, arguments: Any) -> Dict[str, Any]:
    if not isinstance(arguments, dict):
        raise TypeError(
            f"'{predicate_name}' predicate must define 'arguments' as a dict. "
            f"Got {type(arguments)}."
        )
    return dict(arguments)


def validate_goal_arguments(
    predicate_name: str,
    arguments: Any,
    goal_cls: type[BaseGoal],
) -> Dict[str, Any]:
    arguments = validate_predicate_arguments(predicate_name, arguments)

    supported_parameters: List[str] = []
    required_parameters: List[str] = []

    for parameter_name, parameter in inspect.signature(goal_cls.__init__).parameters.items():
        if parameter_name == "self":
            continue
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            continue
        supported_parameters.append(parameter_name)
        if parameter.default is inspect.Signature.empty:
            required_parameters.append(parameter_name)

    extra_keys = set(arguments) - set(supported_parameters)
    if extra_keys:
        extra_key = sorted(extra_keys)[0]
        raise TypeError(
            f"'{predicate_name}' predicate does not support argument {extra_key!r}. "
            f"Supported arguments are {sorted(supported_parameters)}."
        )

    missing_keys = [key for key in required_parameters if key not in arguments]
    if missing_keys:
        raise ValueError(
            f"'{predicate_name}' predicate is missing required arguments {missing_keys}."
        )

    return arguments


def validate_custom_arguments(
    predicate_name: str,
    arguments: Any,
    required_keys: Tuple[str, ...],
    optional_keys: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    arguments = validate_predicate_arguments(predicate_name, arguments)
    allowed_keys = set(required_keys) | set(optional_keys)

    for raw_key in arguments:
        if raw_key not in allowed_keys:
            raise TypeError(
                f"'{predicate_name}' predicate does not support argument {raw_key!r}. "
                f"Supported arguments are {sorted(allowed_keys)}."
            )

    missing_keys = [key for key in required_keys if key not in arguments]
    if missing_keys:
        raise ValueError(
            f"'{predicate_name}' predicate is missing required arguments {missing_keys}."
        )

    return arguments


def validate_in_arguments(arguments: Any) -> Dict[str, Any]:
    arguments = validate_goal_arguments("in", arguments, At)
    validate_entity_selector(arguments["obj_name"], "'in' predicate obj_name")
    validate_entity_selector(arguments["location"], "'in' predicate location")
    if "thresh" in arguments:
        validate_threshold(arguments["thresh"], "'in' predicate thresh")
    return arguments


def validate_not_in_arguments(arguments: Any) -> Dict[str, Any]:
    arguments = validate_goal_arguments("not_in", arguments, NotAt)
    validate_entity_selector(arguments["obj_name"], "'not_in' predicate obj_name")
    validate_entity_selector(arguments["locations"], "'not_in' predicate locations")
    if not isinstance(arguments["strict"], bool):
        raise TypeError(
            f"'not_in' predicate strict must be a bool. Got {type(arguments['strict'])}."
        )
    if "thresh" in arguments:
        validate_threshold(arguments["thresh"], "'not_in' predicate thresh")
    return arguments


def validate_on_arguments(arguments: Any) -> Dict[str, Any]:
    arguments = validate_goal_arguments("on", arguments, On)
    validate_entity_selector(arguments["top_object"], "'on' predicate top_object")
    validate_entity_selector(arguments["bottom_object"], "'on' predicate bottom_object")
    if "thresh" in arguments:
        validate_threshold(arguments["thresh"], "'on' predicate thresh")
    return arguments


def validate_on_mutual_arguments(arguments: Any) -> Dict[str, Any]:
    arguments = validate_custom_arguments(
        "on_mutual",
        arguments,
        required_keys=("objects_1", "objects_2"),
        optional_keys=("thresh",),
    )
    validate_entity_selector(arguments["objects_1"], "'on_mutual' predicate objects_1")
    validate_entity_selector(arguments["objects_2"], "'on_mutual' predicate objects_2")
    if "thresh" in arguments:
        validate_threshold(arguments["thresh"], "'on_mutual' predicate thresh")
    return arguments


def validate_count_arguments(arguments: Any) -> Dict[str, Any]:
    arguments = validate_goal_arguments("count", arguments, AtLeastCountAt)
    validate_entity_selector(arguments["objects"], "'count' predicate objects")
    validate_entity_selector(arguments["location"], "'count' predicate location")
    if not isinstance(arguments["minimum"], int):
        raise TypeError(f"'count' minimum must be an int, got {type(arguments['minimum'])}")
    if "thresh" in arguments:
        validate_threshold(arguments["thresh"], "'count' predicate thresh")
    return arguments


def validate_mug_and_capsule_arguments(arguments: Any) -> Dict[str, Any]:
    arguments = validate_custom_arguments(
        "MugAndCapsuleGoal",
        arguments,
        required_keys=("capsule",),
    )
    validate_string(arguments["capsule"], "'MugAndCapsuleGoal' predicate capsule")
    return arguments


def get_obs_entity_names(obs: Dict[str, Any]) -> List[str]:
    extra = obs.get("extra", None)
    if not isinstance(extra, dict):
        raise RuntimeError("Benchmark predicate evaluation requires obs['extra'] to be a dict.")
    return list(extra.keys())


def get_obs_entity_signature(obs: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(sorted(get_obs_entity_names(obs)))


def expand_entity_arg(entity_names: List[str], value: Any) -> List[str]:
    if isinstance(value, str):
        if "*" in value:
            return star_extractor(entity_names, value)
        return [value]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(expand_entity_arg(entity_names, item))
        return out
    raise RuntimeError(
        f"Error in predicate argument, got type {type(value)} which is supposed "
        "to be either a List[str] or a str"
    )


def combine_goals(goals: List[BaseGoal]) -> BaseGoal:
    if len(goals) == 0:
        raise ValueError("Can not combine an empty list of goals.")
    if len(goals) == 1:
        return goals[0]
    return And(goals)


def build_in(entity_names: List[str], arguments: Dict[str, Any]) -> BaseGoal:
    arguments = validate_in_arguments(arguments)
    objects = expand_entity_arg(entity_names, arguments["obj_name"])
    targets = expand_entity_arg(entity_names, arguments["location"])
    thresh = arguments.get("thresh", None)
    return combine_goals(
        [
            At(obj, target, thresh=thresh) if thresh is not None else At(obj, target)
            for obj, target in product(objects, targets)
        ]
    )


def build_not_in(entity_names: List[str], arguments: Dict[str, Any]) -> BaseGoal:
    arguments = validate_not_in_arguments(arguments)
    objects = expand_entity_arg(entity_names, arguments["obj_name"])
    targets = expand_entity_arg(entity_names, arguments["locations"])
    strict = arguments["strict"]
    thresh = arguments.get("thresh", None)
    return combine_goals(
        [
            NotAt(obj, [target], strict=strict, thresh=thresh)
            if thresh is not None
            else NotAt(obj, [target], strict=strict)
            for obj, target in product(objects, targets)
        ]
    )


def build_on(entity_names: List[str], arguments: Dict[str, Any]) -> BaseGoal:
    arguments = validate_on_arguments(arguments)
    top_objects = expand_entity_arg(entity_names, arguments["top_object"])
    bottom_objects = expand_entity_arg(entity_names, arguments["bottom_object"])
    thresh = arguments.get("thresh", None)
    return combine_goals(
        [
            On(top, bottom, thresh=thresh) if thresh is not None else On(top, bottom)
            for top, bottom in product(top_objects, bottom_objects)
        ]
    )


def build_on_mutual(entity_names: List[str], arguments: Dict[str, Any]) -> BaseGoal:
    arguments = validate_on_mutual_arguments(arguments)
    objects_1 = expand_entity_arg(entity_names, arguments["objects_1"])
    objects_2 = expand_entity_arg(entity_names, arguments["objects_2"])
    thresh = arguments.get("thresh", 0.05)
    return combine_goals(
        [At(obj_1, obj_2, thresh=thresh) for obj_1, obj_2 in product(objects_1, objects_2)]
    )


def build_count(entity_names: List[str], arguments: Dict[str, Any]) -> BaseGoal:
    arguments = validate_count_arguments(arguments)
    objects = expand_entity_arg(entity_names, arguments["objects"])
    targets = expand_entity_arg(entity_names, arguments["location"])
    minimum = arguments["minimum"]
    thresh = arguments.get("thresh", None)
    return combine_goals(
        [
            AtLeastCountAt(objects, target, minimum, thresh=thresh)
            if thresh is not None
            else AtLeastCountAt(objects, target, minimum)
            for target in targets
        ]
    )


predicate_definitions: Dict[str, PredicateDefinition] = {
    "in": PredicateDefinition(
        name="in",
        validator=validate_in_arguments,
        builder=build_in,
    ),
    "not_in": PredicateDefinition(
        name="not_in",
        validator=validate_not_in_arguments,
        builder=build_not_in,
    ),
    "on": PredicateDefinition(
        name="on",
        validator=validate_on_arguments,
        builder=build_on,
    ),
    "on_mutual": PredicateDefinition(
        name="on_mutual",
        validator=validate_on_mutual_arguments,
        builder=build_on_mutual,
    ),
    "count": PredicateDefinition(
        name="count",
        validator=validate_count_arguments,
        builder=build_count,
    ),
}


predicate_argument_validators: Dict[str, PredicateArgumentsValidator] = {
    name: definition.validator for name, definition in predicate_definitions.items()
}


predicate_registry: Dict[str, GoalBuilder] = {
    name: definition.builder for name, definition in predicate_definitions.items()
}
