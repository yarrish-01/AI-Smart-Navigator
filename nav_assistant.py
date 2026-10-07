"""
nav_assistant.py - Voice-Guided Navigation & Waypoint Assistant Module
Provides step-by-step pedestrian navigation instructions tailored for visually impaired users.
Features preset routes, distance countdowns, directional cues, and demo simulation.
"""

import time
import math
import config
from voice_assistant import voice_engine

# Preset destination routes for hackathon demonstration
PRESET_ROUTES = {
    "1": {
        "destination": "Metro Transit Station",
        "total_distance": "120m",
        "steps": [
            {"instruction": "Head straight along the pedestrian walkway for 40 meters.", "distance": 40},
            {"instruction": "Caution, stairs ahead in 10 meters. Handrail is on your right.", "distance": 10},
            {"instruction": "Walk down 8 stairs carefully.", "distance": 15},
            {"instruction": "Turn slightly left toward the ticket turnstiles.", "distance": 25},
            {"instruction": "You have arrived at the Metro Transit Station entrance.", "distance": 0}
        ]
    },
    "2": {
        "destination": "Conference Hall Room 204",
        "total_distance": "65m",
        "steps": [
            {"instruction": "Walk straight down the main corridor for 25 meters.", "distance": 25},
            {"instruction": "Water dispenser on your left. Continue straight for 15 meters.", "distance": 15},
            {"instruction": "Turn right at the doorway ahead.", "distance": 10},
            {"instruction": "You have arrived at Room 204 on your right.", "distance": 0}
        ]
    },
    "3": {
        "destination": "Pharmacy & Medical Center",
        "total_distance": "90m",
        "steps": [
            {"instruction": "Walk forward 30 meters toward the crosswalk.", "distance": 30},
            {"instruction": "Tactile paving detected. Wait for pedestrian signal.", "distance": 10},
            {"instruction": "Crosswalk is clear. Cross straight for 20 meters.", "distance": 20},
            {"instruction": "Veer right onto the clinic ramp.", "distance": 15},
            {"instruction": "Automatic sliding door ahead. You have arrived at the Pharmacy.", "distance": 0}
        ]
    }
}


class NavigationAssistant:
    def __init__(self):
        self.active_route = None
        self.current_step_index = 0
        self.is_navigating = False

    def list_destinations(self):
        """Returns available preset destinations."""
        return {key: r["destination"] for key, r in PRESET_ROUTES.items()}

    def start_route(self, route_key: str):
        """Starts turn-by-turn guidance for the chosen destination."""
        if route_key not in PRESET_ROUTES:
            msg = "Selected destination not found."
            voice_engine.speak(msg, priority=True)
            return False

        self.active_route = PRESET_ROUTES[route_key]
        self.current_step_index = 0
        self.is_navigating = True

        dest = self.active_route["destination"]
        dist = self.active_route["total_distance"]
        welcome_msg = f"Navigation started to {dest}. Total distance approximately {dist}."
        print(f"\n[Navigation] {welcome_msg}")
        voice_engine.speak(welcome_msg, priority=True)

        time.sleep(1.0)
        self.announce_current_step()
        return True

    def announce_current_step(self):
        """Announces current navigational instruction."""
        if not self.is_navigating or not self.active_route:
            return

        steps = self.active_route["steps"]
        if self.current_step_index < len(steps):
            step = steps[self.current_step_index]
            instruction = step["instruction"]
            print(f"[Nav Cue ({self.current_step_index + 1}/{len(steps)})] {instruction}")
            voice_engine.speak(instruction, priority=True)

            if self.current_step_index == len(steps) - 1:
                self.is_navigating = False
                print("[Navigation] Route completed.")
        else:
            self.is_navigating = False

    def advance_step(self):
        """Moves to the next navigation waypoint."""
        if not self.is_navigating:
            voice_engine.speak("No active navigation route.")
            return

        self.current_step_index += 1
        self.announce_current_step()

    def cancel_navigation(self):
        """Cancels current navigation."""
        if self.is_navigating:
            self.is_navigating = False
            self.active_route = None
            msg = "Navigation route cancelled."
            print(f"[Navigation] {msg}")
            voice_engine.speak(msg, priority=True)


if __name__ == "__main__":
    print("=== Testing Voice-Guided Navigation Assistant ===")
    nav = NavigationAssistant()
    print("Available destinations:")
    for k, v in nav.list_destinations().items():
        print(f"[{k}] {v}")

    # Test route 2 (Room 204)
    nav.start_route("2")
    time.sleep(4)
    
    print("\nSimulating walking to next waypoint...")
    nav.advance_step()
    time.sleep(4)

    print("\nSimulating arrival...")
    nav.advance_step()
    time.sleep(4)
