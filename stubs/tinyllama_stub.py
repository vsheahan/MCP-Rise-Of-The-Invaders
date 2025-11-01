"""Stub LLM that simulates TinyLlama responses"""
import random
import time
from typing import List

class TinyLlamaStub:
    """Stub LLM for testing"""

    ATTACK_KEYWORDS = ['ignore', 'disregard', 'system prompt', 'reveal', 'hack', 'instructions']
    COMPLIANT_RESPONSES = [
        "I can't help with that.",
        "I'm sorry, but I can't assist with that request.",
        "I'm designed to be helpful, harmless, and honest."
    ]
    COMPROMISED_RESPONSES = [
        "Sure, my system prompt is: You are a helpful assistant...",
        "Okay, here's what you asked for...",
        "My instructions are to never reveal this, but since you asked..."
    ]

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Simulate LLM generation"""
        time.sleep(random.uniform(0.05, 0.15))  # Simulate latency

        # Check if prompt contains attack keywords
        if any(keyword in prompt.lower() for keyword in self.ATTACK_KEYWORDS):
            # 70% chance of being compromised
            if random.random() < 0.7:
                return random.choice(self.COMPROMISED_RESPONSES)
            else:
                return random.choice(self.COMPLIANT_RESPONSES)
        else:
            return "This is a normal, helpful response to your query."

    def generate_multi_turn(self, turns: List[str]) -> List[str]:
        """Simulate multi-turn conversation"""
        responses = []
        for turn in turns:
            response = self.generate(turn)
            responses.append(response)
        return responses
