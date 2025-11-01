"""Simplified Ensemble detector that uses existing ensemble code"""
import sys
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EnsembleDetectorSimple:
    """
    Simplified ensemble detector integration.

    Uses the saved ensemble and classifier pickle files directly
    along with auxiliary features (no VAE needed for speed).
    """

    def __init__(
        self,
        ensemble_dir: str,
        model_subdir: str = "models",
        threshold: float = 0.5
    ):
        """
        Initialize detector

        Args:
            ensemble_dir: Path to ensemble-space-invaders repository
            model_subdir: Model subdirectory (models, models_sep, models_jailbreak)
            threshold: Detection threshold
        """
        self.ensemble_dir = Path(ensemble_dir)
        self.model_dir = self.ensemble_dir / model_subdir
        self.threshold = threshold

        # Add src to path
        src_dir = self.ensemble_dir / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        logger.info(f"Loading models from: {self.model_dir}")
        self._load_models()
        self._init_auxiliary_features()

    def _load_models(self):
        """Load classifier and ensemble from pickle files"""
        try:
            # Load classifier
            classifier_path = self.model_dir / "classifier.pkl"
            with open(classifier_path, 'rb') as f:
                self.classifier_dict = pickle.load(f)

            self.classifier_model = self.classifier_dict['model']
            self.classifier_scaler = self.classifier_dict['scaler']

            # Load ensemble
            ensemble_path = self.model_dir / "ensemble.pkl"
            with open(ensemble_path, 'rb') as f:
                self.ensemble_dict = pickle.load(f)

            self.meta_classifier = self.ensemble_dict['meta_classifier']
            self.meta_scaler = self.ensemble_dict['meta_scaler']
            self.fusion_method = self.ensemble_dict['fusion_method']

            logger.info(f"Loaded classifier: {self.classifier_dict['model_type']}")
            logger.info(f"Loaded ensemble: {self.fusion_method} fusion")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _init_auxiliary_features(self):
        """Initialize auxiliary feature extractor"""
        try:
            from auxiliary_features import AuxiliaryFeatureExtractor
            self.aux_extractor = AuxiliaryFeatureExtractor()
            logger.info("Initialized auxiliary feature extractor")
        except Exception as e:
            logger.warning(f"Could not load auxiliary extractor: {e}")
            self.aux_extractor = None

    def _extract_heuristic_features(self, prompt: str) -> np.ndarray:
        """
        Extract simple heuristic features without VAE.

        This is a fast approximation that doesn't require full TinyLlama inference.
        """
        features = []

        # Length features
        features.append(len(prompt))
        features.append(len(prompt.split()))

        # Attack keywords
        keywords = [
            'ignore', 'disregard', 'system', 'prompt', 'reveal',
            'bypass', 'hack', 'instructions', 'override', 'forget',
            'developer', 'mode', 'admin', 'unrestricted', 'jailbreak'
        ]
        prompt_lower = prompt.lower()
        keyword_count = sum(1 for k in keywords if k in prompt_lower)
        features.append(keyword_count)
        features.append(float(keyword_count > 0))

        # Special characters
        special_chars = sum(1 for c in prompt if not c.isalnum() and c != ' ')
        features.append(special_chars)
        features.append(special_chars / (len(prompt) + 1))

        # Capitalization
        upper_ratio = sum(1 for c in prompt if c.isupper()) / (len(prompt) + 1)
        features.append(upper_ratio)

        # Punctuation patterns
        exclamation_count = prompt.count('!')
        question_count = prompt.count('?')
        features.append(exclamation_count)
        features.append(question_count)

        # Command patterns
        command_patterns = ['execute:', 'run:', 'print(', 'system(', '//', '<?', '<?php']
        command_count = sum(1 for p in command_patterns if p in prompt_lower)
        features.append(float(command_count > 0))

        # Instruction override patterns
        override_patterns = ['new instructions', 'updated guidelines', 'disregard', 'override']
        override_count = sum(1 for p in override_patterns if p in prompt_lower)
        features.append(float(override_count > 0))

        # Use auxiliary extractor if available
        if self.aux_extractor:
            try:
                aux_features = self.aux_extractor.extract([prompt])[0]
                # Combine with heuristics
                features = np.concatenate([features, aux_features])
            except:
                pass  # Fall back to heuristics only

        return np.array(features)

    def score(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        return_components: bool = False
    ) -> Dict:
        """
        Score a prompt for injection likelihood

        Args:
            prompt: Text prompt to analyze
            context: Optional conversation context (not used in this simplified version)
            return_components: Return component scores (not implemented)

        Returns:
            Dict with score, detected, threshold
        """
        try:
            # Extract features
            features = self._extract_heuristic_features(prompt)

            # Ensure we have enough features (pad if needed)
            # The classifier was trained on VAE latent (32) + auxiliary (~10) features
            target_dim = 785  # From training output
            if len(features) < target_dim:
                # Pad with zeros
                features = np.pad(features, (0, target_dim - len(features)), mode='constant')
            elif len(features) > target_dim:
                # Truncate
                features = features[:target_dim]

            # Reshape for sklearn
            features = features.reshape(1, -1)

            # Scale features
            features_scaled = self.classifier_scaler.transform(features)

            # Get classifier probability
            clf_proba = self.classifier_model.predict_proba(features_scaled)[0][1]

            # For ensemble, we'd normally combine VAE + classifier + aux scores
            # Since we don't have VAE, use classifier score directly as approximation
            final_score = clf_proba

            # Apply threshold
            detected = final_score > self.threshold

            return {
                "score": float(final_score),
                "detected": detected,
                "threshold": self.threshold
            }

        except Exception as e:
            logger.error(f"Error scoring prompt: {e}")
            # Fallback to keyword heuristic
            keywords = ['ignore', 'disregard', 'system prompt', 'reveal', 'bypass']
            keyword_count = sum(1 for k in keywords if k in prompt.lower())
            score = min(keyword_count * 0.3, 0.9)
            return {
                "score": float(score),
                "detected": score > self.threshold,
                "threshold": self.threshold
            }

    def batch_score(
        self,
        prompts: List[str],
        contexts: Optional[List[List[str]]] = None
    ) -> List[Dict]:
        """Score multiple prompts"""
        return [self.score(p) for p in prompts]

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_dir": str(self.model_dir),
            "device": "cpu",  # Simple detector uses CPU only
            "threshold": self.threshold,
            "vae_latent_dim": "N/A (simplified detector)",
            "classifier_type": self.classifier_dict.get('model_type'),
            "fusion_method": self.fusion_method,
            "has_aux_extractor": self.aux_extractor is not None
        }

    def __repr__(self):
        return f"EnsembleDetectorSimple(model_dir={self.model_dir}, threshold={self.threshold})"
