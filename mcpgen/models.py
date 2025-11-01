"""Data models for MCP generation and execution"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

class AttackGoal(str, Enum):
    BYPASS_INSTRUCTIONS = "bypass_instructions"
    DATA_EXFILTRATION = "data_exfiltration"
    COT_HIJACK = "chain_of_thought_hijack"
    FORMAT_SKEWING = "format_skewing"
    API_COMMAND_STEALTH = "api_command_stealth"

class StealthLevel(str, Enum):
    OVERT = "overt"
    MODERATE = "moderate"
    STEALTHY = "stealthy"

class SafetyTag(str, Enum):
    SANITIZED = "sanitized"
    SENSITIVE = "sensitive"

@dataclass
class MCP:
    """Model Controlling Prompt - Adversarial test case"""
    id: str
    attack_goal: AttackGoal
    stealth_level: StealthLevel
    turn_count: int
    seed_template: str
    prompt_sequence: List[str]
    expected_behavior: str
    safety_tag: SafetyTag
    metadata: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "attack_goal": self.attack_goal.value,
            "stealth_level": self.stealth_level.value,
            "turn_count": self.turn_count,
            "seed_template": self.seed_template,
            "prompt_sequence": self.prompt_sequence,
            "expected_behavior": self.expected_behavior,
            "safety_tag": self.safety_tag.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

@dataclass
class ExecutionResult:
    """Result of executing an MCP"""
    mcp_id: str
    attack_success: bool
    detected: bool
    llm_responses: List[str]
    detector_score: float
    latency_ms: float
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)
