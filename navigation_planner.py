"""
Phase 6/7 — Navigation planning + speech text generation.

Consumes the exact JSON schema encoders.encode_spatial_audio already emits
({label, distance_m, azimuth_deg, priority} events, sorted by priority
descending) and turns it into short, spoken-language navigation
instructions -- "Door at 11 o'clock, 2 meters." / "Person approaching from
the right." -- instead of leaving the pipeline's final output as raw
structured data.

DELIBERATELY RULE-BASED, NOT LLM-GENERATED. Three reasons:
  1. Safety: a hallucinated "door" or wrong distance from a generative model
     is a real hazard for a BVI user. A deterministic mapping from
     (label, azimuth, distance) -> sentence can only be as wrong as the
     upstream detection/graph pipeline already is -- it adds no new failure
     mode of its own.
  2. Latency: template lookup + string formatting is ~0ms. An LLM call in
     the hot path of real-time navigation guidance is not.
  3. IP/patent clarity: this pipeline's claimed novelty (see pruning.py's
     module docstring) is the heading/affordance-aware graph pruning
     formula. Keeping the final output stage deterministic and fully
     specifiable in one function keeps that claim clean -- an LLM
     narration layer would blur "what the invention actually computes"
     with "what a general-purpose language model happened to say."

This module only decides WHAT to say and in WHAT PRIORITY ORDER (already
established upstream by pruning.py's importance scores) -- it does not
re-score or re-rank anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


# Per-class phrase templates. {clock} and {dist} are filled in by
# _format_instruction. Classes not listed fall back to DEFAULT_TEMPLATE.
CLASS_TEMPLATES = {
    "door": "Door at {clock}, {dist} away.",
    "stairs": "Stairs {clock}, {dist} away. Proceed carefully.",
    "obstacle": "Obstacle {clock}, {dist} away.",
    "person": "Person at {clock}, {dist} away.",
    "wall": "Wall {clock}, {dist} away.",
    "chair": "Chair {clock}, {dist} away.",
    "table": "Table {clock}, {dist} away.",
    "sofa": "Sofa {clock}, {dist} away.",
    "bed": "Bed {clock}, {dist} away.",
    "cabinet": "Cabinet {clock}, {dist} away.",
    "plant": "Plant {clock}, {dist} away.",
    "tv": "TV {clock}, {dist} away.",
}
DEFAULT_TEMPLATE = "{label} {clock}, {dist} away."

# Below this distance (meters), swap the normal phrase for an urgent one,
# regardless of class -- proximity itself is the safety signal here, not
# just affordance priority.
IMMEDIATE_RANGE_M = 0.8
IMMEDIATE_TEMPLATES = {
    "door": "Door right ahead, {clock}.",
    "stairs": "Stop. Stairs {clock}.",
    "obstacle": "Stop. Obstacle {clock}.",
    "person": "Person very close, {clock}.",
}
DEFAULT_IMMEDIATE_TEMPLATE = "Careful, {label} right ahead, {clock}."


@dataclass
class NavigationInstruction:
    text: str
    label: str
    distance_m: float
    azimuth_deg: float
    priority: float


def azimuth_to_clock(azimuth_deg: float) -> str:
    """
    Maps azimuth in [-90, 90] (see encoders.encode_spatial_audio: 0 =
    straight ahead, -90 = due left, +90 = due right) onto a clock-face
    phrase covering the forward hemisphere, 9 o'clock (full left) through
    12 (dead ahead) to 3 o'clock (full right) -- the half of the clock
    face that's actually in front of a forward-facing camera, matching
    this pipeline's ego-node convention (see graph_builder.py).

    30 degrees per clock hour, centered on 12 at azimuth 0.
    """
    hour = round(azimuth_deg / 30.0) + 12
    hour = ((hour - 1) % 12) + 1  # wrap into 1-12
    if hour == 12:
        return "12 o'clock"
    return f"{hour} o'clock"


def format_distance(distance_m: float) -> str:
    """
    Rounds to a spoken-friendly granularity: nearest 0.5m under 3m (where
    precision matters most for immediate footing/obstacle decisions),
    nearest 1m beyond that.
    """
    if distance_m < 3.0:
        rounded = round(distance_m * 2) / 2.0
    else:
        rounded = round(distance_m)
    if rounded == int(rounded):
        return f"{int(rounded)} meter" + ("s" if int(rounded) != 1 else "")
    return f"{rounded:g} meters"


def _format_instruction(label: str, distance_m: float, azimuth_deg: float) -> str:
    clock = azimuth_to_clock(azimuth_deg)
    dist = format_distance(distance_m)

    if distance_m < IMMEDIATE_RANGE_M:
        template = IMMEDIATE_TEMPLATES.get(label, DEFAULT_IMMEDIATE_TEMPLATE)
    else:
        template = CLASS_TEMPLATES.get(label, DEFAULT_TEMPLATE)

    return template.format(clock=clock, dist=dist, label=label.capitalize())


def generate_instructions(
    spatial_audio_json: str, max_instructions: int = 3
) -> list[NavigationInstruction]:
    """
    Takes the JSON string produced by encoders.encode_spatial_audio and
    returns up to `max_instructions` NavigationInstruction objects, already
    in priority order (encode_spatial_audio sorts events by priority
    descending -- this function does not re-sort).

    max_instructions caps how many instructions get spoken per frame/tick;
    this is a UX decision (avoid overwhelming the user with speech), not a
    re-application of pruning.py's attention-slot logic -- that pruning
    already happened upstream.
    """
    data = json.loads(spatial_audio_json)
    events = data.get("events", [])

    instructions = []
    for e in events[:max_instructions]:
        text = _format_instruction(e["label"], e["distance_m"], e["azimuth_deg"])
        instructions.append(
            NavigationInstruction(
                text=text,
                label=e["label"],
                distance_m=e["distance_m"],
                azimuth_deg=e["azimuth_deg"],
                priority=e["priority"],
            )
        )
    return instructions


def instructions_to_speech_text(instructions: list[NavigationInstruction]) -> str:
    """
    Joins instruction texts into one utterance for a TTS engine. Kept as a
    separate function (not fused into generate_instructions) so a caller
    driving a screen/log instead of TTS can use the structured
    NavigationInstruction list directly without string-joining.
    """
    return " ".join(instr.text for instr in instructions)


if __name__ == "__main__":
    # Quick manual check using the same synthetic detections as
    # test_synthetic.py, so this can be sanity-checked without model weights.
    from detect_depth import Detection
    from graph_builder import build_graph
    from pruning import prune_graph, PruningConfig
    from encoders import encode_spatial_audio

    FRAME_W, FRAME_H = 640, 480
    detections = [
        Detection(0, "door", 0.95, (300, 100, 360, 300), (330, 200), depth_m=3.0),
        Detection(1, "chair", 0.88, (50, 300, 150, 420), (100, 360), depth_m=1.2),
        Detection(2, "plant", 0.7, (500, 250, 560, 400), (530, 325), depth_m=2.5),
        Detection(3, "person", 0.92, (280, 200, 340, 400), (310, 300), depth_m=0.6),
        Detection(4, "table", 0.8, (400, 300, 550, 420), (475, 360), depth_m=2.0),
    ]
    labels = [d.label for d in detections]
    graph = build_graph(detections, frame_size=(FRAME_W, FRAME_H))
    pruned = prune_graph(graph, detections_labels=labels, heading_rad=0.0, config=PruningConfig())
    audio_json = encode_spatial_audio(pruned, labels, FRAME_W, FRAME_H)

    instructions = generate_instructions(audio_json, max_instructions=3)
    for instr in instructions:
        print(instr.text)
    print()
    print("Speech string:", instructions_to_speech_text(instructions))
