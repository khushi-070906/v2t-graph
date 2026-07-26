"""
Live camera loop, tracked version — the "best" version: same Phases 1-7 as
live_camera_loop.py, plus Phase 5.5 (tracker.py) sitting between detection
and the graph/pruning pipeline.

What this fixes vs. the stateless live_camera_loop.py:
    - No flicker: a detection must be confirmed for `confirm_frames`
      consecutive frames before it can be spoken, and survives up to
      `max_missed_frames` of not being seen before being dropped. A
      borderline chair popping in and out at the confidence threshold no
      longer produces a new spoken instruction every time it blinks.
    - Real "approaching" phrasing: navigation_planner's per-frame distance
      snapshot is now backed by tracker.py's measured depth_velocity_mps,
      so "person approaching" means the tracker actually measured the
      person's distance decreasing over multiple frames -- not a guess
      from one frame.

What did NOT change: detect_depth.py, graph_builder.py, pruning.py,
encoders.py are untouched. The tracker only gates WHICH detections reach
the graph-building step (only confirmed tracks) and adds one motion
annotation on top of navigation_planner's output text -- it does not
re-score priority or change pruning.py's formula. If pruning correctness
ever needs debugging, live_camera_loop.py (untracked) isolates that from
any tracker-layer bugs.

Usage:
    python live_camera_loop_tracked.py --detector-weights yolov8s-worldv2.pt --conf 0.15 --device cpu
"""

from __future__ import annotations

import argparse
import time

import cv2

from detect_depth import DetectorDepthEstimator
from graph_builder import build_graph
from pruning import prune_graph, PruningConfig
from encoders import encode_spatial_audio
from navigation_planner import generate_instructions, instructions_to_speech_text
from tracker import ObjectTracker


def draw_overlay(frame, instruction_lines: list[str]) -> None:
    y = 30
    for line in instruction_lines:
        cv2.putText(
            frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2, cv2.LINE_AA,
        )
        y += 28


def annotate_approaching(instructions, detections, det_to_track, tracker) -> list[str]:
    """
    Appends " Approaching." to an instruction's text when the underlying
    detection has a confirmed track with measured closing velocity.
    Matches instructions back to detections by (label, distance_m) --
    approximate but sufficient here since generate_instructions() already
    only received this frame's confirmed detections (see run_camera_loop).
    """
    lines = []
    for instr in instructions:
        text = instr.text
        for i, det in enumerate(detections):
            if det.label != instr.label:
                continue
            if abs(det.depth_m - instr.distance_m) > 1e-3:
                continue
            track = det_to_track.get(i)
            if track is not None and tracker.is_approaching(track):
                text = text.rstrip(".") + ". Approaching."
            break
        lines.append(text)
    return lines


def run_camera_loop(
    camera_index: int = 0,
    heading_rad: float = 0.0,
    device: str = "cpu",
    detector_weights: str = "yolov10n.pt",
    conf_threshold: float = 0.35,
    max_instructions: int = 3,
    speak: bool = True,
    speak_interval_sec: float = 2.5,
    display: bool = True,
    confirm_frames: int = 3,
    max_missed_frames: int = 5,
) -> None:
    print("Loading detector + depth models (once, reused every frame)...")
    detector = DetectorDepthEstimator(
        device=device, detector_weights=detector_weights, conf_threshold=conf_threshold
    )
    tracker = ObjectTracker(confirm_frames=confirm_frames, max_missed_frames=max_missed_frames)

    speech_engine = None
    if speak:
        from speech_output import SpeechEngine  # lazy import, pyttsx3 optional
        speech_engine = SpeechEngine()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    last_speak_time = 0.0
    print("Tracked camera loop running. Press 'q' in the display window to quit (Ctrl+C if headless).")
    print(f"(Detections need {confirm_frames} consecutive frames before they're spoken.)")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed, skipping.")
                continue

            h, w = frame.shape[:2]

            # Phase 1 — raw per-frame detections (unfiltered)
            raw_detections = detector.run(frame)

            # Phase 5.5 — track across frames, get back only CONFIRMED
            # detections (i.e. seen `confirm_frames` times in a row)
            det_to_track = tracker.update(raw_detections, w, h)
            confirmed_detections = [raw_detections[i] for i in det_to_track.keys()]
            # re-key det_to_track against the filtered/re-indexed list so
            # annotate_approaching's index lookup stays valid
            confirmed_det_to_track = {
                new_i: det_to_track[old_i]
                for new_i, old_i in enumerate(det_to_track.keys())
            }
            labels = [d.label for d in confirmed_detections]

            # Phases 2-4 — identical to pipeline.py, just fed only
            # confirmed detections instead of every raw detection
            graph = build_graph(confirmed_detections, frame_size=(w, h))
            pruned = prune_graph(
                graph, detections_labels=labels, heading_rad=heading_rad, config=PruningConfig()
            )
            audio_json = encode_spatial_audio(pruned, labels, w, h)

            # Phase 6 — instructions, then annotate with measured
            # "approaching" from the tracker (does not touch priority/order)
            instructions = generate_instructions(audio_json, max_instructions=max_instructions)
            instruction_lines = annotate_approaching(
                instructions, confirmed_detections, confirmed_det_to_track, tracker
            )

            now = time.time()
            if speak and instruction_lines and (now - last_speak_time) >= speak_interval_sec:
                speech_engine.speak(" ".join(instruction_lines))
                last_speak_time = time.time()

            if display:
                draw_overlay(frame, instruction_lines)
                cv2.imshow("V-to-T Graph — live (tracked)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif instruction_lines:
                print(" | ".join(instruction_lines))

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--heading", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detector-weights", default="yolov10n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--max-instructions", type=int, default=3)
    parser.add_argument("--no-speak", action="store_true")
    parser.add_argument("--speak-interval", type=float, default=2.5)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--confirm-frames", type=int, default=3, help="consecutive frames before a detection is trusted/spoken")
    parser.add_argument("--max-missed-frames", type=int, default=5, help="grace period before a track is dropped")
    args = parser.parse_args()

    run_camera_loop(
        camera_index=args.camera,
        heading_rad=args.heading,
        device=args.device,
        detector_weights=args.detector_weights,
        conf_threshold=args.conf,
        max_instructions=args.max_instructions,
        speak=not args.no_speak,
        speak_interval_sec=args.speak_interval,
        display=not args.no_display,
        confirm_frames=args.confirm_frames,
        max_missed_frames=args.max_missed_frames,
    )
