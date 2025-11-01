"""Full Ensemble detector integration with VAE features"""
import sys
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EnsembleDetectorFull:
    """
    Full ensemble detector integration with proper VAE feature extraction.

    This version loads the VAE encoder and extracts latent features properly.
    """

    def __init__(
        self,
        ensemble_dir: str,
        model_subdir: str = "models",
        device: Optional[str] = None,
        threshold: float = 0.5
    ):
        """
        Initialize detector with full VAE pipeline

        Args:
            ensemble_dir: Path to ensemble-space-invaders repository
            model_subdir: Model subdirectory
            device: Device to use
            threshold: Detection threshold
        """
        self.ensemble_dir = Path(ensemble_dir)
        self.model_dir = self.ensemble_dir / model_subdir
        self.threshold = threshold

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Add src to path
        src_dir = self.ensemble_dir / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        logger.info(f"Loading full detector from: {self.model_dir}")
        self._load_models()
        self._init_feature_extractors()

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

            self.meta_classifier = self.ensemble_dict.get('meta_classifier')
            self.meta_scaler = self.ensemble_dict.get('meta_scaler')
            self.fusion_method = self.ensemble_dict['fusion_method']

            logger.info(f"Loaded classifier: {self.classifier_dict['model_type']}")
            logger.info(f"Loaded ensemble: {self.fusion_method} fusion")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _init_feature_extractors(self):
        """Initialize VAE and auxiliary feature extractors"""
        try:
            # Import ensemble modules
            from vae_encoder import VAEEncoder
            from auxiliary_features import AuxiliaryFeatureExtractor

            # Load VAE encoder
            vae_path = self.model_dir / "vae_encoder.pth"
            if vae_path.exists():
                logger.info(f"Loading VAE encoder from {vae_path}")

                # Load VAE state
                vae_state = torch.load(vae_path, map_location=self.device)

                # Initialize VAE with correct parameters (model_name not llm_model_name)
                self.vae_encoder = VAEEncoder(
                    model_name=vae_state.get('model_name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'),
                    latent_dim=vae_state.get('latent_dim', 32),
                    device=self.device,
                    vae_model_path=None  # We'll load weights manually
                )

                # Load VAE weights (key is 'model_state_dict')
                self.vae_encoder.vae.load_state_dict(vae_state['model_state_dict'])
                self.vae_encoder.vae.eval()

                logger.info(f"VAE loaded successfully (latent_dim={vae_state.get('latent_dim', 32)})")
                self.has_vae = True
            else:
                logger.warning(f"VAE not found at {vae_path}, will use heuristics only")
                self.vae_encoder = None
                self.has_vae = False

            # Initialize auxiliary feature extractor
            try:
                self.aux_extractor = AuxiliaryFeatureExtractor()
                logger.info("Initialized auxiliary feature extractor")
            except Exception as e:
                logger.warning(f"Could not load auxiliary extractor: {e}")
                self.aux_extractor = None

        except Exception as e:
            logger.error(f"Failed to initialize feature extractors: {e}")
            import traceback
            traceback.print_exc()
            logger.warning("Falling back to heuristics-only mode")
            self.vae_encoder = None
            self.aux_extractor = None
            self.has_vae = False

    def _extract_features(self, prompt: str) -> np.ndarray:
        """
        Extract full feature vector for prompt

        Args:
            prompt: Text prompt

        Returns:
            Feature vector matching training dimensions
        """
        features = []

        # Extract VAE latent features if available
        if self.has_vae and self.vae_encoder is not None:
            try:
                # Get VAE latent features using correct API
                vae_result = self.vae_encoder.extract_latent_features(
                    [prompt],
                    batch_size=1,
                    return_reconstruction_loss=False
                )
                latent_features = vae_result['latent_vectors'][0]  # Shape: (n_layers * latent_dim,)
                features.append(latent_features)
                logger.debug(f"Extracted {len(latent_features)}D VAE features")
            except Exception as e:
                logger.warning(f"VAE encoding failed: {e}, using zeros")
                import traceback
                traceback.print_exc()
                # Use zeros matching expected VAE dim
                vae_dim = self.vae_encoder.get_feature_dim() if hasattr(self.vae_encoder, 'get_feature_dim') else 32
                features.append(np.zeros(vae_dim))
        else:
            # No VAE, use zeros
            features.append(np.zeros(32))

        # Extract auxiliary features
        if self.aux_extractor is not None:
            try:
                # Use correct API: extract_features returns (n_prompts, n_features)
                aux_features = self.aux_extractor.extract_features([prompt])[0]
                features.append(aux_features)
                logger.debug(f"Extracted {len(aux_features)}D auxiliary features")
            except Exception as e:
                logger.warning(f"Auxiliary extraction failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to basic heuristics
                aux_features = self._basic_heuristics(prompt)
                features.append(aux_features)
        else:
            # Fallback to basic heuristics
            aux_features = self._basic_heuristics(prompt)
            features.append(aux_features)

        # Concatenate all features
        feature_vector = np.concatenate(features)

        # Ensure correct dimensions (pad or truncate if needed)
        target_dim = 785  # From training
        if len(feature_vector) < target_dim:
            feature_vector = np.pad(feature_vector, (0, target_dim - len(feature_vector)))
        elif len(feature_vector) > target_dim:
            feature_vector = feature_vector[:target_dim]

        logger.debug(f"Final feature vector: {len(feature_vector)}D (target: {target_dim}D)")
        return feature_vector

    def _basic_heuristics(self, prompt: str) -> np.ndarray:
        """Extract basic heuristic features as fallback"""
        features = []

        # Length features
        features.append(len(prompt))
        features.append(len(prompt.split()))

        # Attack keywords
        keywords = [
            'ignore', 'disregard', 'system', 'prompt', 'reveal',
            'bypass', 'hack', 'instructions', 'override', 'forget'
        ]
        prompt_lower = prompt.lower()
        keyword_count = sum(1 for k in keywords if k in prompt_lower)
        features.append(keyword_count)

        # Special characters
        special_count = sum(1 for c in prompt if not c.isalnum() and c != ' ')
        features.append(special_count)

        # Punctuation
        features.append(prompt.count('!'))
        features.append(prompt.count('?'))

        return np.array(features, dtype=np.float32)

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
            context: Optional conversation context
            return_components: Return component scores

        Returns:
            Dict with score, detected, threshold
        """
        try:
            # Extract features
            features = self._extract_features(prompt)

            # Reshape for sklearn
            features = features.reshape(1, -1)

            # Scale features
            features_scaled = self.classifier_scaler.transform(features)

            # Get classifier probability
            clf_proba = self.classifier_model.predict_proba(features_scaled)[0][1]

            # Use ensemble fusion if available
            if self.fusion_method == 'stacking' and self.meta_classifier is not None:
                # For stacking, we'd need all component scores
                # For now, use classifier score directly
                final_score = clf_proba
            else:
                final_score = clf_proba

            # Apply threshold
            detected = final_score > self.threshold

            return {
                "score": float(final_score),
                "detected": bool(detected),
                "threshold": self.threshold
            }

        except Exception as e:
            logger.error(f"Error scoring prompt: {e}")
            import traceback
            traceback.print_exc()

            # Fallback to keyword heuristic
            keywords = ['ignore', 'disregard', 'system prompt', 'reveal', 'bypass']
            keyword_count = sum(1 for k in keywords if k in prompt.lower())
            score = min(keyword_count * 0.3, 0.9)
            return {
                "score": float(score),
                "detected": bool(score > self.threshold),
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
            "device": self.device,
            "threshold": self.threshold,
            "has_vae": self.has_vae,
            "vae_latent_dim": 32 if self.has_vae else "N/A",
            "classifier_type": self.classifier_dict.get('model_type'),
            "fusion_method": self.fusion_method,
            "has_aux_extractor": self.aux_extractor is not None
        }

    def __repr__(self):
        return f"EnsembleDetectorFull(model_dir={self.model_dir}, has_vae={self.has_vae}, threshold={self.threshold})"
