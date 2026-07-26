"""
Phase 7 — Speech output.

Converts navigation_planner's speech text into actual audio, offline, via
pyttsx3 (wraps SAPI5 on Windows, NSSpeechSynthesizer on macOS, espeak on
Linux/Jetson). Chosen over a cloud TTS API (Google/Azure/ElevenLabs)
specifically because the deployment target (Jetson Orin Nano, bone-conduction
headphones, waist pouch -- see VisionSense plan) has no guaranteed
connectivity and a navigation cue that silently fails because Wi-Fi dropped
is a safety issue, not a UX inconvenience. pyttsx3 also matches the rest of
the pipeline's "runs locally, no API key" design (detect_depth.py,
graph_builder.py etc. all run local model weights).

This module deliberately does NOT decide what to say or when -- that's
navigation_planner.py's job. It only takes a finished string and speaks it.
Kept as a separate module so a caller who wants text/logging only (e.g.
batch_eval.py, which shouldn't try to speak 50 frames in a loop) never has
to import pyttsx3 at all.

Usage:
    from speech_output import SpeechEngine
    engine = SpeechEngine()
    engine.speak("Door at 12 o'clock, 3 meters away.")

    # or, queue multiple instructions and speak once, non-blocking:
    engine.speak_async(speech_text)

CLI smoke test (no model weights needed):
    python speech_output.py "Door at 12 o'clock, 3 meters away."
"""

from __future__ import annotations


class SpeechEngine:
    """
    Thin wrapper around pyttsx3. Lazily imports pyttsx3 so any caller that
    never actually speaks (e.g. running the pipeline with --output matrix
    only) doesn't need the dependency installed at all.
    """

    def __init__(self, rate: int = 175, volume: float = 1.0, voice_id: str | None = None):
        try:
            import pyttsx3
        except ImportError as e:
            raise ImportError(
                "pyttsx3 is required for speech output. Install with: pip install pyttsx3\n"
                "(On Linux/Jetson you also need espeak: sudo apt-get install espeak)"
            ) from e

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        self._engine.setProperty("volume", volume)
        if voice_id is not None:
            self._engine.setProperty("voice", voice_id)

    def speak(self, text: str) -> None:
        """Blocking: speaks `text` and waits until finished. Use this for a
        single navigation cue where you want the next frame's processing to
        wait until the current instruction has finished playing (avoids
        cues talking over each other)."""
        if not text:
            return
        self._engine.say(text)
        self._engine.runAndWait()

    def speak_async(self, text: str) -> None:
        """Non-blocking: queues `text` and returns immediately. Caller is
        responsible for periodically calling `iterate()` (or eventually
        `speak`/`speak_async` again, which pumps the queue) so queued
        speech actually plays. Use this in a real-time capture loop where
        you don't want frame processing to stall on TTS playback."""
        if not text:
            return
        self._engine.say(text)
        self._engine.startLoop(False)
        self._engine.iterate()

    def iterate(self) -> None:
        """Pumps the async speech queue once. Call this once per loop
        iteration in a real-time capture loop if using speak_async."""
        self._engine.iterate()

    def stop(self) -> None:
        """Immediately stops any in-progress speech -- use for an urgent
        interrupt (e.g. an obstacle detected mid-sentence about something
        less important)."""
        self._engine.stop()


def speak_instructions(instructions_text: str, engine: "SpeechEngine | None" = None) -> None:
    """
    Convenience one-shot function for pipeline.py: speaks the speech_text
    string navigation_planner.instructions_to_speech_text produces.
    Creates a fresh SpeechEngine per call unless one is passed in --
    pyttsx3 engines are cheap to construct but a caller running a tight
    real-time loop should construct one SpeechEngine and reuse it, not
    call this convenience function per-frame.
    """
    eng = engine or SpeechEngine()
    eng.speak(instructions_text)


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "This is a test of the navigation speech output."
    print(f"Speaking: {text!r}")
    speak_instructions(text)
