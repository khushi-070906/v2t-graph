"""
Live camera loop — Phases 1-7 run continuously on a webcam feed instead of
a single static image path.

DELIBERATELY STATELESS. Each frame is processed completely independently:
fresh detection, fresh graph, fresh pruning, fresh instructions. Nothing
carries over from the previous frame — no object tracking, no ID
persistence, no smoothing/debouncing of instructions across frames. This
is intentional, not a missing feature: the temporal/tracking layer
("person approaching" as an actual velocity estimate, object permanence)
is flagged separately in README's next-steps as its own future phase.
Bolting partial state into this loop would blur that boundary and make it
harder to reason about what's a per-frame pipeline result vs. what's
tracking-layer logic later.

Practical consequence of being stateless: instructions can flicker/change
between frames if a detection is borderline (e.g. a chair right at the
confidence threshold appearing and disappearing frame to frame). That's
expected here and is exactly the kind of thing a future tracking layer
would smooth out -- not something this module tries to paper over.

Usage:
    python live_camera_loop.py --detector-weights yolov8s-worldv2.pt --conf 0.15 --device cpu
    python live_camera_loop.py --camera 0 --heading 0.0 --speak-interval 2.0 --no-display

Press 'q' in the display window (or Ctrl+C in headless mode) to quit.
"""

from __future__ import annotations

import argparse
import time

import cv2

from detect_depth import DetectorDepthEstimator
from graph_builder import build_graph
from pruning import prune_graph, PruningConfig, load_pruning_config
from encoders import encode_spatial_audio
from navigation_planner import generate_instructions, instructions_to_speech_text


def draw_overlay(frame, instructions_text: list[str]) -> None:
    """Draws the current frame's instruction lines onto the display frame,
    in-place. Debug/demo aid only -- not part of the pipeline output."""
    y = 30
    for line in instructions_text:
        cv2.putText(
            frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2, cv2.LINE_AA,
        )
        y += 28


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
    calibration_path: str | None = None,
    pruning_config_path: str | None = None,
) -> None:
    print("Loading detector + depth models (once, reused every frame)...")
    detector = DetectorDepthEstimator(
        device=device, detector_weights=detector_weights, conf_threshold=conf_threshold
    )

    calibration = None
    calibration_warned_classes = set()
    apply_calibration_to_detections = None
    if calibration_path is not None:
        from calibrate_depth import load_calibration, apply_calibration_to_detections
        calibration = load_calibration(calibration_path)
        print(f"Loaded depth calibration from {calibration_path}: "
              f"calibrated classes = {list(calibration.keys())}")

    pruning_config, affordance_priority = load_pruning_config(pruning_config_path)

    speech_engine = None
    if speak:
        from speech_output import SpeechEngine  # lazy import, pyttsx3 optional
        speech_engine = SpeechEngine()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    last_speak_time = 0.0
    print("Camera loop running. Press 'q' in the display window to quit (Ctrl+C if headless).")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed, skipping.")
                continue

            h, w = frame.shape[:2]

            # Phases 1-4 -- identical single-frame pipeline as pipeline.py,
            # just called in a loop instead of once. No state passed between
            # iterations (see module docstring).
            detections = detector.run(frame)
            if calibration is not None:
                # apply_calibration_to_detections was imported once, above
                # the loop, alongside load_calibration -- previously this
                # import ran on every single frame; module re-imports are
                # cheap but it's still a needless dict lookup + attribute
                # bind per frame in the hot path for no benefit.
                detections = apply_calibration_to_detections(detections, calibration, calibration_warned_classes)
            labels = [d.label for d in detections]
            graph = build_graph(detections, frame_size=(w, h))
            pruned = prune_graph(
                graph, detections_labels=labels, heading_rad=heading_rad,
                config=pruning_config, affordance_priority=affordance_priority,
            )
            audio_json = encode_spatial_audio(pruned, labels, w, h)

            # Phase 6 -- always compute instructions (cheap, deterministic),
            # independent of whether we actually speak this tick.
            instructions = generate_instructions(audio_json, max_instructions=max_instructions)
            instruction_lines = [instr.text for instr in instructions]

            # Phase 7 -- rate-limited speech. TTS playback (~1-3s) is far
            # slower than the detection loop, so speaking every frame would
            # queue up a backlog of stale instructions. speak_interval_sec
            # throttles how often a NEW utterance is spoken; it does not
            # affect how often detection/instructions are recomputed above.
            now = time.time()
            if speak and instructions and (now - last_speak_time) >= speak_interval_sec:
                speech_text = instructions_to_speech_text(instructions)
                speech_engine.speak(speech_text)  # blocking -- see module docstring
                last_speak_time = time.time()

            if display:
                draw_overlay(frame, instruction_lines)
                cv2.imshow("V-to-T Graph — live", frame)
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
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--heading", type=float, default=0.0, help="user heading in radians, 0 = facing right in image plane")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--detector-weights", default="yolov10n.pt",
        help='e.g. "yolov8s-worldv2.pt" for open-vocabulary detection (see DetectorDepthEstimator docstring)',
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--max-instructions", type=int, default=3)
    parser.add_argument("--no-speak", action="store_true", help="disable TTS, print instructions instead")
    parser.add_argument("--speak-interval", type=float, default=2.5, help="minimum seconds between spoken utterances")
    parser.add_argument("--no-display", action="store_true", help="headless mode (Jetson without a monitor) — prints instructions instead of showing a window")
    parser.add_argument("--calibration", default=None, help="path to calibration.json from calibrate_depth.py")
    parser.add_argument("--pruning-config", default=None, help="path to a JSON file (see pruning_config.example.json) overriding tau/heading_gamma/prune_threshold/max_nodes/affordance_priority")
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
        calibration_path=args.calibration,
        pruning_config_path=args.pruning_config,
    )
