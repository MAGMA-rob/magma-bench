from typing import Dict, List

from ._log_rule_compiler import LogRuleCompiler
from ._log_rule_registry import (
    BaseLogRule,
    CountLogRule,
    NotInLogRule,
    RequiresBeforeLogRule,
    RuleBuilder,
    log_matches,
    log_rule_registry,
    validate_event_matcher,
)


_log_rule_compiler = LogRuleCompiler(log_rule_registry)

__all__ = [
    "BaseLogRule",
    "CountLogRule",
    "NotInLogRule",
    "RequiresBeforeLogRule",
    "RuleBuilder",
    "compile_log_rules",
    "log_matches",
    "log_rule_registry",
    "validate_event_matcher",
]


def compile_log_rules(rule_specs: List[Dict]) -> List[BaseLogRule]:
    return _log_rule_compiler.compile_many(rule_specs)
