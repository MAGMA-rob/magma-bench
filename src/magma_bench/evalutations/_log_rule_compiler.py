from typing import Any, Dict, List

from ._log_rule_registry import BaseLogRule, RuleBuilder


class LogRuleCompiler:
    def __init__(self, registry: Dict[str, RuleBuilder]) -> None:
        self.registry = dict(registry)

    def compile_rule(self, rule_spec: Dict[str, Any], context: str) -> BaseLogRule:
        if not isinstance(rule_spec, dict):
            raise TypeError(f"{context} must be a dict. Got {type(rule_spec)}.")

        extra_keys = set(rule_spec) - {"rule_name", "arguments"}
        if extra_keys:
            raise TypeError(
                f"{context} only supports keys ['arguments', 'rule_name']. "
                f"Got unexpected keys {sorted(extra_keys)}."
            )

        rule_name = rule_spec.get("rule_name")
        if not isinstance(rule_name, str) or rule_name == "":
            raise ValueError(f"{context} must define a non-empty string in 'rule_name'.")

        builder = self.registry.get(rule_name)
        if builder is None:
            raise ValueError(
                f"{context} uses unknown rule_name {rule_name!r}. "
                f"Known rules are {sorted(self.registry)}."
            )

        arguments = rule_spec.get("arguments")
        if not isinstance(arguments, dict):
            raise TypeError(
                f"{context} must define 'arguments' as a dict. Got {type(arguments)}."
            )

        try:
            return builder(arguments)
        except Exception as exc:
            raise type(exc)(f"{context}: {exc}") from exc

    def compile_many(self, rule_specs: List[Dict[str, Any]]) -> List[BaseLogRule]:
        return [
            self.compile_rule(rule_spec, context=f"log_rules[{idx}]")
            for idx, rule_spec in enumerate(rule_specs)
        ]
