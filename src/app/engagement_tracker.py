# src/app/engagement_tracker.py

import collections

class EngagementTracker:
    """
    Keeps a sliding window of the last N emotions
    and computes a simple engagement score.
    """

    def __init__(self, window_size=30):
        self.window = collections.deque(maxlen=window_size)

    def add_observation(self, emotion):
        self.window.append(emotion)

    def get_engagement_score(self):
        if not self.window:
            return 0.0

        score_map = {
            "Happy": 2.0,
            "Surprise": 2.0,
            "Neutral": 1.0,
            "Sad": -1.0,
            "Angry": -2.0,
            "Fear": -1.5,
            "Disgust": -1.0,
        }

        total = sum(score_map.get(e, 0) for e in self.window)
        return total / len(self.window)
