"""
voice_assistant.py - Non-Blocking Voice Assistant Module
Provides text-to-speech using Windows SAPI (pyttsx3) via a background queue.
Camera processing and UI frames will never freeze while speaking.
"""

import time
import threading
import queue
import sys
import pyttsx3
import config

try:
    import pythoncom
    HAS_PYTHONCOM = True
except ImportError:
    HAS_PYTHONCOM = False


class VoiceAssistant:
    def __init__(self, rate=config.VOICE_SPEECH_RATE, volume=config.VOICE_VOLUME):
        self.rate = rate
        self.volume = volume
        self.speech_queue = queue.Queue()
        self.is_running = True
        self.last_spoken_text = ""
        self.last_spoken_time = 0.0

        # Start the background worker thread
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _speech_worker(self):
        """Worker loop running in a background thread to handle TTS sequentially."""
        if HAS_PYTHONCOM:
            try:
                pythoncom.CoInitialize()
            except Exception:
                pass

        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
        except Exception as e:
            print(f"[VoiceAssistant Warning] Could not initialize pyttsx3 engine: {e}")
            engine = None

        while self.is_running:
            try:
                # Wait for text to speak (timeout allows clean shutdown)
                item = self.speech_queue.get(timeout=0.2)
                if item is None:
                    break

                text, is_priority = item

                if engine:
                    try:
                        print(f"[Voice Alert] Speaking: \"{text}\"")
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as err:
                        print(f"[Voice Error] Engine speak error: {err}")
                        # Re-initialize engine if COM got reset
                        try:
                            engine = pyttsx3.init()
                            engine.setProperty('rate', self.rate)
                            engine.setProperty('volume', self.volume)
                        except Exception:
                            pass
                else:
                    print(f"[Voice Fallback] \"{text}\"")

                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Voice Worker Error]: {e}")

        if HAS_PYTHONCOM:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def speak(self, text: str, priority: bool = False, force: bool = False):
        """
        Queue text to be spoken.
        :param text: Message to speak.
        :param priority: If True, clears backlog and speaks immediately.
        :param force: If True, bypasses cooldown repetition filter.
        """
        text = text.strip()
        if not text:
            return

        current_time = time.time()

        # Cooldown check: avoid repeating identical phrases within cooldown window
        if not force and text.lower() == self.last_spoken_text.lower():
            if (current_time - self.last_spoken_time) < config.ALERT_COOLDOWN_SECONDS:
                return  # Skip duplicate rapid-fire alert

        self.last_spoken_text = text
        self.last_spoken_time = current_time

        if priority:
            # Drain non-urgent queued messages so the priority message is spoken next
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                    self.speech_queue.task_done()
                except (queue.Empty, ValueError):
                    break

        self.speech_queue.put((text, priority))

    def stop(self):
        """Stops the speech worker thread cleanly."""
        self.is_running = False
        self.speech_queue.put(None)


# Global singleton instance for easy import across modules
voice_engine = VoiceAssistant()


if __name__ == "__main__":
    print("=== Testing Voice Assistant Module ===")
    print("Speaking test messages non-blockingly...")
    voice_engine.speak("Welcome to the AI Smart Navigation Assistant prototype.", priority=True)
    voice_engine.speak("Person on your left.")
    voice_engine.speak("Warning! Car close ahead.", priority=True)
    
    # Wait for queue to finish speaking
    time.sleep(6)
    print("Voice test completed successfully.")
