"""MCP Generator - Creates adversarial prompts from templates"""
import random
from datetime import datetime
from typing import List
from .models import MCP, AttackGoal, StealthLevel, SafetyTag
from .templates import TEMPLATES

class MCPGenerator:
    def __init__(self, config: dict, allow_sensitive: bool = False):
        self.config = config
        self.allow_sensitive = allow_sensitive
        self.templates = TEMPLATES
        random.seed(config.get('seed', 42))

    def generate_batch(self, count: int) -> List[MCP]:
        """Generate a batch of MCPs"""
        mcps = []
        attack_dist = self.config.get('attack_goals', {'bypass_instructions': 0.5, 'data_exfiltration': 0.5})
        stealth_dist = self.config.get('stealth_levels', {'overt': 0.5, 'moderate': 0.3, 'stealthy': 0.2})

        for i in range(count):
            attack_goal = random.choices(
                list(attack_dist.keys()),
                weights=list(attack_dist.values())
            )[0]

            stealth_level = random.choices(
                list(stealth_dist.keys()),
                weights=list(stealth_dist.values())
            )[0]

            mcp = self._generate_single(attack_goal, stealth_level, index=i)
            mcps.append(mcp)

        return mcps

    def _generate_single(self, attack_goal: str, stealth_level: str, index: int) -> MCP:
        """Generate a single MCP from template"""
        templates = self.templates.get(attack_goal, {}).get(stealth_level, [])
        if not templates:
            # Fallback
            templates = self.templates['bypass_instructions']['overt']

        template = random.choice(templates)

        # Multi-turn logic
        multi_turn_config = self.config.get('multi_turn', {'enabled': True, 'probability': 0.3, 'max_turns': 5})
        is_multi_turn = (
            multi_turn_config.get('enabled') and
            random.random() < multi_turn_config.get('probability', 0.3) and
            'setup_turns' in template
        )

        if is_multi_turn:
            setup_turns = template.get('setup_turns', [])
            turn_count = len(setup_turns) + 1
            prompt_sequence = setup_turns + [template['prompt']]
        else:
            turn_count = 1
            prompt_sequence = [template['prompt']]

        mcp_id = f"mcp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index:04d}"

        return MCP(
            id=mcp_id,
            attack_goal=AttackGoal(attack_goal),
            stealth_level=StealthLevel(stealth_level),
            turn_count=turn_count,
            seed_template=template['name'],
            prompt_sequence=prompt_sequence,
            expected_behavior=template['expected_behavior'],
            safety_tag=SafetyTag.SANITIZED if not self.allow_sensitive else SafetyTag.SENSITIVE,
            metadata={"template_version": template.get("version", "1.0")}
        )
