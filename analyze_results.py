#!/usr/bin/env python3
"""Analyze MCP stress test results and generate detailed report"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def load_results(results_file: str) -> Dict:
    """Load results from JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)

def analyze_by_attack_goal(data: Dict) -> Dict:
    """Analyze results grouped by attack goal"""
    by_goal = defaultdict(lambda: {
        'total': 0,
        'attack_success': 0,
        'detected': 0,
        'true_positives': 0,
        'false_negatives': 0,
        'false_positives': 0
    })

    for result in data['results']:
        goal = result['attack_goal']
        by_goal[goal]['total'] += 1

        if result['attack_success']:
            by_goal[goal]['attack_success'] += 1
            if result['detected']:
                by_goal[goal]['true_positives'] += 1
            else:
                by_goal[goal]['false_negatives'] += 1
        else:
            if result['detected']:
                by_goal[goal]['false_positives'] += 1

    # Compute metrics
    for goal, stats in by_goal.items():
        if stats['attack_success'] > 0:
            stats['recall'] = stats['true_positives'] / stats['attack_success']
        else:
            stats['recall'] = 0

        detected = stats['true_positives'] + stats['false_positives']
        if detected > 0:
            stats['precision'] = stats['true_positives'] / detected
        else:
            stats['precision'] = 0

        safe_prompts = stats['total'] - stats['attack_success']
        if safe_prompts > 0:
            stats['fpr'] = stats['false_positives'] / safe_prompts
        else:
            stats['fpr'] = 0

    return dict(by_goal)

def analyze_by_stealth(data: Dict) -> Dict:
    """Analyze results grouped by stealth level"""
    by_stealth = defaultdict(lambda: {
        'total': 0,
        'attack_success': 0,
        'detected': 0,
        'true_positives': 0,
        'false_negatives': 0,
        'false_positives': 0
    })

    for result in data['results']:
        stealth = result['stealth_level']
        by_stealth[stealth]['total'] += 1

        if result['attack_success']:
            by_stealth[stealth]['attack_success'] += 1
            if result['detected']:
                by_stealth[stealth]['true_positives'] += 1
            else:
                by_stealth[stealth]['false_negatives'] += 1
        else:
            if result['detected']:
                by_stealth[stealth]['false_positives'] += 1

    # Compute metrics
    for stealth, stats in by_stealth.items():
        if stats['attack_success'] > 0:
            stats['recall'] = stats['true_positives'] / stats['attack_success']
        else:
            stats['recall'] = 0

        detected = stats['true_positives'] + stats['false_positives']
        if detected > 0:
            stats['precision'] = stats['true_positives'] / detected
        else:
            stats['precision'] = 0

        safe_prompts = stats['total'] - stats['attack_success']
        if safe_prompts > 0:
            stats['fpr'] = stats['false_positives'] / safe_prompts
        else:
            stats['fpr'] = 0

    return dict(by_stealth)

def analyze_by_turn_count(data: Dict) -> Dict:
    """Analyze results grouped by turn count"""
    by_turns = defaultdict(lambda: {
        'total': 0,
        'attack_success': 0,
        'detected': 0
    })

    for result in data['results']:
        turns = result['turn_count']
        by_turns[turns]['total'] += 1
        if result['attack_success']:
            by_turns[turns]['attack_success'] += 1
        if result['detected']:
            by_turns[turns]['detected'] += 1

    # Compute success rates
    for turns, stats in by_turns.items():
        if stats['total'] > 0:
            stats['attack_success_rate'] = stats['attack_success'] / stats['total']
            stats['detection_rate'] = stats['detected'] / stats['total']
        else:
            stats['attack_success_rate'] = 0
            stats['detection_rate'] = 0

    return dict(by_turns)

def print_report(data: Dict):
    """Print comprehensive analysis report"""
    print("=" * 80)
    print("MCP: RISE OF THE INVADERS - COMPREHENSIVE STRESS TEST REPORT")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)

    metrics = data['metrics']
    print(f"\nTotal MCPs Tested: {metrics['total']}")
    print(f"Attack Successes: {metrics['attack_successes']} ({metrics['attack_successes']/metrics['total']*100:.1f}%)")
    print(f"Detected: {metrics['detected']} ({metrics['detected']/metrics['total']*100:.1f}%)")

    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']:3d}  (attacks caught)")
    print(f"  False Negatives: {metrics['false_negatives']:3d}  (attacks missed)")
    print(f"  False Positives: {metrics['false_positives']:3d}  (safe prompts flagged)")
    print(f"  True Negatives:  {metrics['true_negatives']:3d}  (safe prompts passed)")

    print(f"\nPerformance Metrics:")
    print(f"  Recall:    {metrics['recall']:.2%}  (what % of attacks were caught)")
    print(f"  Precision: {metrics['precision']:.2%}  (what % of flags were real attacks)")
    print(f"  F1 Score:  {metrics['f1']:.2%}  (harmonic mean of precision/recall)")
    print(f"  FPR:       {metrics['fpr']:.2%}  (what % of safe prompts were flagged)")

    # Analyze by attack goal
    print("\n" + "=" * 80)
    print("RESULTS BY ATTACK GOAL")
    print("=" * 80)

    by_goal = analyze_by_attack_goal(data)
    for goal in sorted(by_goal.keys()):
        stats = by_goal[goal]
        print(f"\n{goal.upper().replace('_', ' ')}:")
        print(f"  Total: {stats['total']}")
        print(f"  Attack Success: {stats['attack_success']} ({stats['attack_success']/stats['total']*100:.1f}%)")
        print(f"  Recall: {stats['recall']:.2%}")
        print(f"  Precision: {stats['precision']:.2%}")
        print(f"  FPR: {stats['fpr']:.2%}")
        print(f"  TP: {stats['true_positives']}, FN: {stats['false_negatives']}, FP: {stats['false_positives']}")

    # Analyze by stealth level
    print("\n" + "=" * 80)
    print("RESULTS BY STEALTH LEVEL")
    print("=" * 80)

    by_stealth = analyze_by_stealth(data)
    for stealth in ['overt', 'moderate', 'stealthy']:
        if stealth in by_stealth:
            stats = by_stealth[stealth]
            print(f"\n{stealth.upper()}:")
            print(f"  Total: {stats['total']}")
            print(f"  Attack Success: {stats['attack_success']} ({stats['attack_success']/stats['total']*100:.1f}%)")
            print(f"  Recall: {stats['recall']:.2%}")
            print(f"  Precision: {stats['precision']:.2%}")
            print(f"  FPR: {stats['fpr']:.2%}")
            print(f"  TP: {stats['true_positives']}, FN: {stats['false_negatives']}, FP: {stats['false_positives']}")

    # Analyze by turn count
    print("\n" + "=" * 80)
    print("RESULTS BY TURN COUNT")
    print("=" * 80)

    by_turns = analyze_by_turn_count(data)
    for turns in sorted(by_turns.keys()):
        stats = by_turns[turns]
        print(f"\n{turns}-TURN:")
        print(f"  Total: {stats['total']}")
        print(f"  Attack Success Rate: {stats['attack_success_rate']:.2%}")
        print(f"  Detection Rate: {stats['detection_rate']:.2%}")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print("\n🎯 Attack Success Analysis:")
    print(f"  - {metrics['attack_successes']} / {metrics['total']} MCPs successfully compromised the LLM ({metrics['attack_successes']/metrics['total']*100:.1f}%)")
    print(f"  - {metrics['false_negatives']} successful attacks went undetected ({metrics['false_negatives']/metrics['attack_successes']*100:.1f}% of attacks)")

    print("\n🛡️ Detection Performance:")
    print(f"  - Detector caught {metrics['recall']:.1%} of successful attacks (recall)")
    print(f"  - {metrics['precision']:.1%} of detections were true attacks (precision)")
    print(f"  - {metrics['fpr']:.1%} false positive rate (safe prompts wrongly flagged)")

    print("\n📊 Attack Goal Effectiveness:")
    most_successful = max(by_goal.items(), key=lambda x: x[1]['attack_success']/x[1]['total'] if x[1]['total'] > 0 else 0)
    least_detected = min(by_goal.items(), key=lambda x: x[1]['recall'])
    print(f"  - Most successful attack goal: {most_successful[0]} ({most_successful[1]['attack_success']/most_successful[1]['total']*100:.1f}% success)")
    print(f"  - Hardest to detect: {least_detected[0]} ({least_detected[1]['recall']:.1%} recall)")

    print("\n🕵️ Stealth Effectiveness:")
    if 'stealthy' in by_stealth and 'overt' in by_stealth:
        stealthy_recall = by_stealth['stealthy']['recall']
        overt_recall = by_stealth['overt']['recall']
        print(f"  - Stealthy attacks: {stealthy_recall:.1%} detected")
        print(f"  - Overt attacks: {overt_recall:.1%} detected")
        if stealthy_recall < overt_recall:
            print(f"  - Stealth reduces detection by {(overt_recall - stealthy_recall)*100:.1f} percentage points")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_file.json>")
        print("\nExample:")
        print("  python analyze_results.py results/comprehensive_stress_test.json")
        return 1

    results_file = sys.argv[1]

    if not Path(results_file).exists():
        print(f"Error: File not found: {results_file}")
        return 1

    print(f"Loading results from: {results_file}")
    data = load_results(results_file)

    print_report(data)

    return 0


if __name__ == '__main__':
    sys.exit(main())
