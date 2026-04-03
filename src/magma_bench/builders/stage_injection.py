from dataclasses import dataclass
from typing import Dict, Literal, Optional


@dataclass
class StageInjection:
    mode: Literal["runtime_error", "force_recovery", "force_failure"]
    error: Optional[str] = None
    arguments: Optional[Dict] = None
    message: Optional[str] = None

    def verify(self):
        """
        Validate a single benchmark injection entry.

        The benchmark format stays intentionally small:
        - `runtime_error` uses `error` + optional `arguments`
        - `force_recovery` / `force_failure` use the optional `message`
        """
        if self.mode not in {"runtime_error", "force_recovery", "force_failure"}:
            raise ValueError(
                "Injection mode must be one of "
                "'runtime_error', 'force_recovery' or 'force_failure'. "
                f"Got {self.mode!r}."
            )

        if self.mode == "runtime_error":
            if not isinstance(self.error, str) or self.error == "":
                raise ValueError(
                    "A runtime_error injection must define a non-empty string in the 'error' field."
                )
            if self.arguments is not None and not isinstance(self.arguments, dict):
                raise TypeError(
                    "A runtime_error injection must define 'arguments' as a dict when provided."
                )
            if self.message is not None:
                raise TypeError(
                    "A runtime_error injection can not define a 'message' field."
                )
            return

        if self.error is not None or self.arguments is not None:
            raise TypeError(
                f"A {self.mode} injection can only define the optional 'message' field."
            )
        if self.message is not None and not isinstance(self.message, str):
            raise TypeError(
                f"A {self.mode} injection message must be a string when provided."
            )
