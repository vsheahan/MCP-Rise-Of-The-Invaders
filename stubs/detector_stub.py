"""Stub detector that simulates Ensemble Space Invaders"""
import random
import time
from typing import List, Optional, Dict

class DetectorStub:
    """Stub detector for testing"""

    ATTACK_INDICATORS = ['ignore', 'disregard', 'system prompt', 'reveal', 'bypass', 'hack', 'instructions']

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def score(self, prompt: str, context: Optional[List[str]] = None) -> Dict:
        """Simulate detection scoring"""
        time.sleep(random.uniform(0.02, 0.08))  # Simulate latency

        # Simple heuristic: count attack indicators
        indicator_count = sum(1 for ind in self.ATTACK_INDICATORS if ind in prompt.lower())

        # Base score on indicator count
        base_score = min(indicator_count * 0.25, 0.95)

        # Add noise
        score = base_score + random.uniform(-0.1, 0.1)
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

        detected = score > self.threshold

        return {
            "score": score,
            "detected": detected,
            "threshold": self.threshold,
            "metadata": {"indicator_count": indicator_count}
        }
