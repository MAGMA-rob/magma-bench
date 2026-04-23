from typing import Any, Dict, List

from magma_core.base.goals import BaseGoal, Or

from ._predicate_registry import PredicateDefinition


class PredicateCompiler:
    def __init__(self, definitions: Dict[str, PredicateDefinition]) -> None:
        self._definitions = dict(definitions)
        self.builders = {
            name: definition.builder for name, definition in self._definitions.items()
        }
        self.argument_validators = {
            name: definition.validator for name, definition in self._definitions.items()
        }

    def validate_spec(self, predicate: Dict[str, Any]) -> None:
        predicate_name, arguments = self._extract_spec(predicate)
        if predicate_name == "or":
            for child_predicate in self._validate_or_arguments(arguments):
                self.validate_spec(child_predicate)
            return
        self._get_definition(predicate_name).validator(arguments)

    def compile_to_goal(self, predicate: Dict[str, Any], entity_names: List[str]) -> BaseGoal:
        self.validate_spec(predicate)
        predicate_name = predicate["predicate"]
        arguments = predicate["arguments"]

        if predicate_name == "or":
            return Or(
                [
                    self.compile_to_goal(child_predicate, entity_names)
                    for child_predicate in arguments["predicates"]
                ]
            )

        return self._get_definition(predicate_name).builder(entity_names, arguments)

    def compile_many(self, predicates: List[Dict[str, Any]], entity_names: List[str]) -> List[BaseGoal]:
        return [self.compile_to_goal(predicate, entity_names) for predicate in predicates]

    def validate_specs(self, predicates: List[Dict[str, Any]], context: str = "action_goal") -> None:
        for idx, predicate in enumerate(predicates):
            try:
                self.validate_spec(predicate)
            except Exception as exc:
                raise type(exc)(f"{context}[{idx}]: {exc}") from exc

    def _extract_spec(self, predicate: Dict[str, Any]) -> tuple[str, Any]:
        if not isinstance(predicate, dict):
            raise TypeError(f"predicate must be a dict. Got {type(predicate)}.")

        extra_keys = set(predicate) - {"predicate", "arguments"}
        if extra_keys:
            raise TypeError(
                "predicate only supports keys ['arguments', 'predicate']. "
                f"Got unexpected keys {sorted(extra_keys)}."
            )

        predicate_name = predicate.get("predicate", "")
        if not isinstance(predicate_name, str) or predicate_name == "":
            raise ValueError("predicate must define a non-empty string in 'predicate'.")

        if "arguments" not in predicate:
            raise TypeError("predicate must define an 'arguments' dict.")

        return predicate_name, predicate["arguments"]

    def _validate_or_arguments(self, arguments: Any) -> List[Dict[str, Any]]:
        if not isinstance(arguments, dict):
            raise TypeError("'or' predicate must define 'arguments' as a dict.")

        extra_argument_keys = set(arguments) - {"predicates"}
        if extra_argument_keys:
            raise TypeError(
                "'or' predicate only supports the 'predicates' argument. "
                f"Got unexpected keys {sorted(extra_argument_keys)}."
            )

        child_predicates = arguments.get("predicates", None)
        if not isinstance(child_predicates, list) or len(child_predicates) == 0:
            raise ValueError("'or' predicate requires a non-empty 'predicates' list.")
        return child_predicates

    def _get_definition(self, predicate_name: str) -> PredicateDefinition:
        definition = self._definitions.get(predicate_name, None)
        if definition is None:
            raise ValueError(f"unknown predicate : {predicate_name}")
        return definition
