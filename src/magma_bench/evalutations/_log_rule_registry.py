from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from magma_core.base.data_structures import Log


RuleBuilder = Callable[[Dict[str, Any]], "BaseLogRule"]

_ALLOWED_EVENT_MATCHER_KEYS = {"function_name", "action", "content"}


def _content_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return list(expected) == list(actual)
    return expected == actual


def validate_event_matcher(matcher: Any, context: str = "event") -> Dict[str, Any]:
    if not isinstance(matcher, dict):
        raise TypeError(f"{context} must be a dict. Got {type(matcher)}.")
    if len(matcher) == 0:
        raise ValueError(f"{context} must not be empty.")

    extra_keys = set(matcher) - _ALLOWED_EVENT_MATCHER_KEYS
    if extra_keys:
        raise TypeError(
            f"{context} only supports keys {sorted(_ALLOWED_EVENT_MATCHER_KEYS)}. "
            f"Got unexpected keys {sorted(extra_keys)}."
        )

    if "function_name" in matcher and not isinstance(matcher["function_name"], str):
        raise TypeError(
            f"{context}['function_name'] must be a str. Got {type(matcher['function_name'])}."
        )
    if "action" in matcher and not isinstance(matcher["action"], str):
        raise TypeError(f"{context}['action'] must be a str. Got {type(matcher['action'])}.")

    return dict(matcher)


def log_matches(log: Log, matcher: Dict[str, Any]) -> bool:
    if "function_name" in matcher and log.function != matcher["function_name"]:
        return False
    if "action" in matcher and log.action != matcher["action"]:
        return False
    if "content" in matcher and not _content_matches(matcher["content"], log.content):
        return False
    return True


class BaseLogRule:
    rule_name: str

    def verify(self, logs: List["Log"]) -> Tuple[bool, str]:
        raise NotImplementedError


@dataclass
class CountLogRule(BaseLogRule):
    event: Dict[str, Any]
    expected_count: int

    rule_name = "count"

    @classmethod
    def from_arguments(cls, arguments: Dict[str, Any]) -> "CountLogRule":
        extra_keys = set(arguments) - {"event", "count"}
        if extra_keys:
            raise TypeError(
                "Rule 'count' only supports keys ['count', 'event']. "
                f"Got unexpected keys {sorted(extra_keys)}."
            )
        if "event" not in arguments or "count" not in arguments:
            raise ValueError("Rule 'count' requires both 'event' and 'count'.")

        expected_count = arguments["count"]
        if not isinstance(expected_count, int):
            raise TypeError(f"Rule 'count' expects an int in 'count'. Got {type(expected_count)}.")
        if expected_count < 0:
            raise ValueError("Rule 'count' expects 'count' to be >= 0.")

        return cls(
            event=validate_event_matcher(arguments["event"], "Rule 'count' event"),
            expected_count=expected_count,
        )

    def verify(self, logs: List["Log"]) -> Tuple[bool, str]:
        current_count = sum(1 for log in logs if log_matches(log, self.event))
        if current_count == self.expected_count:
            return True, ""
        return (
            False,
            f"Log rule 'count' expected {self.expected_count} occurrence(s) of "
            f"{self.event} but got {current_count}.",
        )


@dataclass
class RequiresBeforeLogRule(BaseLogRule):
    event: Dict[str, Any]
    required_before: List[Dict[str, Any]]

    rule_name = "requires_before"

    @classmethod
    def from_arguments(cls, arguments: Dict[str, Any]) -> "RequiresBeforeLogRule":
        extra_keys = set(arguments) - {"event", "required_before"}
        if extra_keys:
            raise TypeError(
                "Rule 'requires_before' only supports keys ['event', 'required_before']. "
                f"Got unexpected keys {sorted(extra_keys)}."
            )
        if "event" not in arguments or "required_before" not in arguments:
            raise ValueError("Rule 'requires_before' requires both 'event' and 'required_before'.")

        required_before = arguments["required_before"]
        if not isinstance(required_before, list):
            raise TypeError(
                "Rule 'requires_before' expects 'required_before' to be a list of event matchers."
            )
        if len(required_before) == 0:
            raise ValueError("Rule 'requires_before' expects a non-empty 'required_before' list.")

        return cls(
            event=validate_event_matcher(arguments["event"], "Rule 'requires_before' event"),
            required_before=[
                validate_event_matcher(matcher, f"Rule 'requires_before' required_before[{idx}]")
                for idx, matcher in enumerate(required_before)
            ],
        )

    def verify(self, logs: List[Log]) -> Tuple[bool, str]:
        event_indices = [
            event_idx
            for event_idx, event_log in enumerate(logs)
            if log_matches(event_log, self.event)
        ]
        if not event_indices:
            return False, "No coffee launched yet"

        event_idx = event_indices[-1]
        previous_event_idx = event_indices[-2] if len(event_indices) > 1 else -1
        # A retry press can reuse earlier setup, but new setup calls between two
        # presses must not contradict the requirement for the latest press.
        event_segment = logs[previous_event_idx + 1:event_idx]

        for matcher in self.required_before:
            if not any(log_matches(previous_log, matcher) for previous_log in logs[:event_idx]):
                return (
                    False,
                    f"Log rule 'requires_before' expected {matcher} before "
                    f"{self.event} at stage log index {event_idx}.",
                )

            function_name = matcher.get("function_name")
            if function_name is None:
                continue

            same_function_logs = [
                (log_idx, log)
                for log_idx, log in enumerate(event_segment, start=previous_event_idx + 1)
                if log.function == function_name
            ]
            if len(same_function_logs) > 1:
                return (
                    False,
                    f"Log rule 'requires_before' expected a single {function_name} "
                    f"before {self.event} at stage log index {event_idx}.",
                )
            if same_function_logs and not log_matches(same_function_logs[0][1], matcher):
                return (
                    False,
                    f"Log rule 'requires_before' expected {matcher} before "
                    f"{self.event}, but found another {function_name} at stage log "
                    f"index {same_function_logs[0][0]}.",
                )
        return True, ""


@dataclass
class NotInLogRule(BaseLogRule):
    event: Dict[str, Any]

    rule_name = "not_in"

    @classmethod
    def from_arguments(cls, arguments: Dict[str, Any]) -> "NotInLogRule":
        extra_keys = set(arguments) - {"event"}
        if extra_keys:
            raise TypeError(
                f"Rule 'not_in' only supports keys ['event']. Got unexpected keys {sorted(extra_keys)}."
            )
        if "event" not in arguments:
            raise ValueError("Rule 'not_in' requires 'event'.")

        return cls(
            event=validate_event_matcher(arguments["event"], "Rule 'not_in' event"),
        )

    def verify(self, logs: List[Log]) -> Tuple[bool, str]:
        for log_idx, log in enumerate(logs):
            if log_matches(log, self.event):
                return (
                    False,
                    f"Log rule 'not_in' expected no occurrence of {self.event} "
                    f"but found one at log index {log_idx}.",
                )
        return True, ""


log_rule_registry: Dict[str, RuleBuilder] = {
    CountLogRule.rule_name: CountLogRule.from_arguments,
    RequiresBeforeLogRule.rule_name: RequiresBeforeLogRule.from_arguments,
    NotInLogRule.rule_name: NotInLogRule.from_arguments,
}
