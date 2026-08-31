from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import textwrap
from typing import Any, Dict, Iterable, Optional


VIDEO_MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class VideoConfig:
    enabled: bool = False
    fps: int = 20
    hold_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("video_fps must be strictly positive")
        if self.hold_seconds <= 0:
            raise ValueError("video_hold_seconds must be strictly positive")


@dataclass
class _VideoSession:
    episode_id: str
    stage_index: int
    instruction: str
    activity_label: str = "Current action"
    activity: str = "Waiting for an agent answer"
    planner_attempt: Optional[int] = None
    planner_max_attempts: int = 10
    planner_message: str = ""
    pending_hold_frames: int = 0
    last_raw_frame: Any = None
    writer: Any = None
    temporary_path: Optional[Path] = None
    final_path: Optional[Path] = None
    frame_count: int = 0
    width: Optional[int] = None
    height: Optional[int] = None


class EpisodeVideoRecorder:
    """Render, annotate and progressively encode one video per active episode."""

    def __init__(self, config: VideoConfig) -> None:
        self.config = config
        self._environment = None
        self._video_directory: Optional[Path] = None
        self._sessions: Dict[int, _VideoSession] = {}
        self._manifest: Dict[str, Any] = {
            "schema_version": VIDEO_MANIFEST_SCHEMA_VERSION,
            "episodes": {},
        }
        self._image_module = None
        self._image_draw_module = None
        self._image_font_module = None
        self._numpy = None
        self._write_frames = None

        if not config.enabled:
            return
        try:
            import numpy
            from PIL import Image, ImageDraw, ImageFont
            from imageio_ffmpeg import write_frames
        except ImportError as error:
            raise RuntimeError(
                "Video recording requires the optional dependencies. Install "
                "magma_bench with `pip install 'magma_bench[video]'` or, from "
                "the repository, `pip install -e '.[video]'`."
            ) from error

        self._numpy = numpy
        self._image_module = Image
        self._image_draw_module = ImageDraw
        self._image_font_module = ImageFont
        self._write_frames = write_frames

    def start_scenario(self, video_directory: Path) -> None:
        if not self.config.enabled:
            return
        if self._sessions:
            raise RuntimeError("Cannot change video scenario while episodes are active")

        self._video_directory = video_directory
        video_directory.mkdir(parents=True, exist_ok=True)
        manifest_path = video_directory / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema_version") != VIDEO_MANIFEST_SCHEMA_VERSION
                or not isinstance(manifest.get("episodes"), dict)
            ):
                raise ValueError(f"Invalid video manifest: {manifest_path}")
            self._manifest = manifest
        else:
            self._manifest = {
                "schema_version": VIDEO_MANIFEST_SCHEMA_VERSION,
                "episodes": {},
            }

    def finish_scenario(self) -> None:
        if not self.config.enabled:
            return
        if self._sessions:
            raise RuntimeError("Cannot finish video scenario while episodes are active")
        self._environment = None
        self._video_directory = None

    def bind_environment(self, environment: Any) -> None:
        if not self.config.enabled:
            return
        if self._sessions:
            raise RuntimeError("Cannot replace the rendered environment while episodes are active")
        self._environment = environment

    def start_episode(
        self,
        env_idx: int,
        episode_id: str,
        stage_index: int,
        instruction_role: str,
        instruction_content: Any,
    ) -> None:
        if not self.config.enabled:
            return
        if self._video_directory is None or self._environment is None:
            raise RuntimeError("Video scenario and environment must be initialized")
        if env_idx in self._sessions:
            raise RuntimeError(f"Environment {env_idx} already records an episode")

        final_path = self._video_directory / f"{episode_id}.mp4"
        temporary_path = self._video_directory / f".{episode_id}.tmp.mp4"
        temporary_path.unlink(missing_ok=True)
        self._sessions[env_idx] = _VideoSession(
            episode_id=episode_id,
            stage_index=stage_index,
            instruction=self._format_instruction(
                instruction_role,
                instruction_content,
            ),
            temporary_path=temporary_path,
            final_path=final_path,
        )

    def update_instruction(
        self,
        env_idx: int,
        stage_index: int,
        role: str,
        content: Any,
        *,
        hold: bool = True,
    ) -> None:
        if not self.config.enabled:
            return
        session = self._sessions[env_idx]
        session.stage_index = stage_index
        session.instruction = self._format_instruction(role, content)
        if hold:
            session.pending_hold_frames += self._hold_frame_count()

    def update_answer(self, env_idx: int, answer: Any, *, hold: bool) -> None:
        if not self.config.enabled:
            return
        session = self._sessions[env_idx]
        say = answer.get_say()
        if say:
            session.activity_label = "Current answer"
            session.activity = str(say)
        else:
            calls = answer.get_action() if hasattr(answer, "get_action") else answer.get_tool_calls()
            lines = []
            for call in calls:
                arguments = ", ".join(
                    f"{name}={value!r}"
                    for name, value in call.arguments.items()
                )
                lines.append(
                    f"{call.target_robot_name}.{call.name}({arguments})"
                )
            session.activity_label = "Current action"
            session.activity = "\n".join(lines) if lines else "No tool call"
        if hold:
            session.pending_hold_frames += self._hold_frame_count()

    def set_planner_retry(
        self,
        env_idx: int,
        attempt: int,
        max_attempts: int,
        message: str,
    ) -> None:
        if not self.config.enabled:
            return
        session = self._sessions.get(env_idx)
        if session is None:
            return
        session.planner_attempt = attempt
        session.planner_max_attempts = max_attempts
        session.planner_message = message
        session.pending_hold_frames += self._hold_frame_count()

    def clear_planner_retry(self, env_idx: int) -> None:
        if not self.config.enabled:
            return
        session = self._sessions.get(env_idx)
        if session is None:
            return
        session.planner_attempt = None
        session.planner_message = ""

    def capture_physical_step(self, env_ids: Iterable[int]) -> None:
        if not self.config.enabled:
            return
        selected = [env_idx for env_idx in env_ids if env_idx in self._sessions]
        if not selected:
            return
        frames = self._render_batch()
        for env_idx in selected:
            raw_frame = self._frame_for_environment(frames, env_idx)
            session = self._sessions[env_idx]
            session.last_raw_frame = raw_frame
            self._write_annotated_frame(session, raw_frame)

    def flush_holds(self, env_ids: Optional[Iterable[int]] = None) -> None:
        if not self.config.enabled:
            return
        selected_ids = (
            set(self._sessions)
            if env_ids is None
            else {env_idx for env_idx in env_ids if env_idx in self._sessions}
        )
        pending_ids = [
            env_idx
            for env_idx in selected_ids
            if self._sessions[env_idx].pending_hold_frames > 0
        ]
        if not pending_ids:
            return

        needs_render = [
            env_idx
            for env_idx in pending_ids
            if self._sessions[env_idx].last_raw_frame is None
        ]
        if needs_render:
            frames = self._render_batch()
            for env_idx in needs_render:
                self._sessions[env_idx].last_raw_frame = self._frame_for_environment(
                    frames,
                    env_idx,
                )

        for env_idx in pending_ids:
            session = self._sessions[env_idx]
            frame = self._compose_frame(session, session.last_raw_frame)
            pending_count = session.pending_hold_frames
            session.pending_hold_frames = 0
            for _ in range(pending_count):
                self._write_frame(session, frame)

    def finish_episode(self, env_idx: int, terminal_status: str) -> None:
        if not self.config.enabled:
            return
        session = self._sessions.get(env_idx)
        if session is None:
            raise RuntimeError(f"Environment {env_idx} has no active video session")

        self.flush_holds([env_idx])
        if session.writer is None:
            frames = self._render_batch()
            raw_frame = self._frame_for_environment(frames, env_idx)
            session.last_raw_frame = raw_frame
            self._write_annotated_frame(session, raw_frame)

        session.writer.close()
        session.writer = None
        if session.temporary_path is None or session.final_path is None:
            raise RuntimeError("Video session paths are not initialized")
        os.replace(session.temporary_path, session.final_path)

        self._manifest["episodes"][session.episode_id] = {
            "file": session.final_path.name,
            "fps": self.config.fps,
            "hold_seconds": self.config.hold_seconds,
            "resolution": [session.width, session.height],
            "frame_count": session.frame_count,
            "duration_seconds": session.frame_count / self.config.fps,
            "terminal_status": terminal_status,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_manifest()
        self._sessions.pop(env_idx)

    def close(self) -> None:
        if not self.config.enabled:
            return
        for session in list(self._sessions.values()):
            if session.writer is not None:
                try:
                    session.writer.close()
                except Exception:
                    pass
            if session.temporary_path is not None:
                session.temporary_path.unlink(missing_ok=True)
        self._sessions.clear()
        self._environment = None
        self._video_directory = None

    def _render_batch(self) -> Any:
        if self._environment is None:
            raise RuntimeError("No environment is bound to the video recorder")
        rendered = self._environment.render_rgb_array()
        if hasattr(rendered, "detach"):
            rendered = rendered.detach()
        if hasattr(rendered, "cpu"):
            rendered = rendered.cpu()
        if hasattr(rendered, "numpy"):
            rendered = rendered.numpy()
        return self._numpy.asarray(rendered)

    def _frame_for_environment(self, frames: Any, env_idx: int) -> Any:
        if frames.ndim == 3:
            if env_idx != 0:
                raise RuntimeError(
                    "The renderer returned one unbatched frame for multiple environments"
                )
            frame = frames
        elif frames.ndim == 4:
            if env_idx >= frames.shape[0]:
                raise RuntimeError(
                    f"Renderer returned {frames.shape[0]} frames for env_idx {env_idx}"
                )
            frame = frames[env_idx]
        else:
            raise RuntimeError(
                f"Unexpected rendered frame shape {getattr(frames, 'shape', None)}"
            )

        if frame.shape[-1] < 3:
            raise RuntimeError(f"Rendered frame must have at least three channels: {frame.shape}")
        frame = frame[..., :3]
        if self._numpy.issubdtype(frame.dtype, self._numpy.floating):
            if frame.size and float(frame.max()) <= 1.0:
                frame = frame * 255.0
            frame = self._numpy.clip(frame, 0, 255)
        return self._numpy.ascontiguousarray(frame.astype(self._numpy.uint8))

    def _write_annotated_frame(self, session: _VideoSession, raw_frame: Any) -> None:
        self._write_frame(session, self._compose_frame(session, raw_frame))

    def _compose_frame(self, session: _VideoSession, raw_frame: Any) -> Any:
        image = self._image_module.fromarray(raw_frame, mode="RGB")
        footer_height = max(240, image.height // 3)
        output_width = image.width + image.width % 2
        output_height = image.height + footer_height
        output_height += output_height % 2
        canvas = self._image_module.new(
            "RGB",
            (output_width, output_height),
            color=(24, 24, 24),
        )
        canvas.paste(image, (0, 0))
        draw = self._image_draw_module.Draw(canvas)
        font_size = max(16, image.width // 32)
        try:
            font = self._image_font_module.truetype("DejaVuSans.ttf", font_size)
            bold_font = self._image_font_module.truetype(
                "DejaVuSans-Bold.ttf",
                font_size,
            )
        except OSError:
            font = self._image_font_module.load_default()
            bold_font = font

        margin = max(12, image.width // 40)
        line_height = max(font_size + 5, 20)
        y = image.height + margin
        draw.text(
            (margin, y),
            f"Stage: {session.stage_index}",
            fill=(255, 255, 255),
            font=bold_font,
        )
        y += line_height + 4

        max_chars = max(28, int((image.width - 2 * margin) / (font_size * 0.58)))
        instruction_lines = textwrap.wrap(
            session.instruction,
            width=max_chars,
            replace_whitespace=True,
        ) or [""]
        activity_lines = []
        for activity_line in session.activity.splitlines() or [""]:
            activity_lines.extend(
                textwrap.wrap(
                    activity_line,
                    width=max_chars,
                    replace_whitespace=True,
                ) or [""]
            )

        available_lines = max(4, (footer_height - (y - image.height) - margin) // line_height)
        instruction_limit = max(2, available_lines // 2)
        activity_limit = max(1, available_lines - instruction_limit - 2)
        instruction_lines = self._truncate_lines(instruction_lines, instruction_limit)
        activity_lines = self._truncate_lines(activity_lines, activity_limit)

        draw.text((margin, y), "Instruction / Status:", fill=(155, 205, 255), font=bold_font)
        y += line_height
        for line in instruction_lines:
            draw.text((margin, y), line, fill=(235, 235, 235), font=font)
            y += line_height
        y += 4
        draw.text((margin, y), f"{session.activity_label}:", fill=(155, 205, 255), font=bold_font)
        y += line_height
        for line in activity_lines:
            draw.text((margin, y), line, fill=(235, 235, 235), font=font)
            y += line_height

        if session.planner_attempt is not None:
            banner_height = max(60, font_size * 3)
            draw.rectangle(
                (0, 0, image.width, banner_height),
                fill=(190, 20, 20),
            )
            banner = (
                "PLANNER ERROR - ENV RETRYING "
                f"(attempt {session.planner_attempt}/{session.planner_max_attempts})"
            )
            draw.text(
                (margin, 8),
                banner,
                fill=(255, 255, 255),
                font=bold_font,
            )
            if session.planner_message:
                planner_message = textwrap.shorten(
                    session.planner_message.replace("\n", " "),
                    width=max_chars,
                    placeholder="…",
                )
                draw.text(
                    (margin, 12 + line_height),
                    planner_message,
                    fill=(255, 230, 230),
                    font=font,
                )

        return self._numpy.asarray(canvas)

    def _write_frame(self, session: _VideoSession, frame: Any) -> None:
        if session.writer is None:
            if session.temporary_path is None:
                raise RuntimeError("Video temporary path is not initialized")
            height, width = frame.shape[:2]
            session.width = width
            session.height = height
            session.writer = self._write_frames(
                str(session.temporary_path),
                (width, height),
                fps=self.config.fps,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                macro_block_size=2,
                ffmpeg_log_level="warning",
                output_params=["-movflags", "+faststart"],
            )
            session.writer.send(None)
        session.writer.send(self._numpy.ascontiguousarray(frame))
        session.frame_count += 1

    def _write_manifest(self) -> None:
        if self._video_directory is None:
            raise RuntimeError("Video directory is not initialized")
        manifest_path = self._video_directory / "manifest.json"
        payload = json.dumps(self._manifest, indent=2, ensure_ascii=False)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._video_directory,
            prefix=".manifest.json.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, manifest_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def _hold_frame_count(self) -> int:
        return max(1, round(self.config.fps * self.config.hold_seconds))

    @staticmethod
    def _format_instruction(role: str, content: Any) -> str:
        if isinstance(content, (dict, list)):
            serialized = json.dumps(content, ensure_ascii=False)
        else:
            serialized = str(content)
        return f"{role}: {serialized}"

    @staticmethod
    def _truncate_lines(lines: list[str], limit: int) -> list[str]:
        if len(lines) <= limit:
            return lines
        truncated = lines[:limit]
        truncated[-1] = truncated[-1].rstrip(" .") + "…"
        return truncated
