"""TinyLlama integration for real LLM inference"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TinyLlamaIntegration:
    """Integration with TinyLlama-1.1B model for local inference"""

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize TinyLlama model

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling vs greedy decoding
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading TinyLlama model: {model_name}")
        logger.info(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        self.model.eval()

        # Configure generation
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        logger.info("TinyLlama loaded successfully")

    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Format prompt for TinyLlama chat model

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt string
        """
        if system_prompt:
            formatted = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        else:
            formatted = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"

        return formatted

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Generate response to a single prompt

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            max_new_tokens: Override default max tokens
            temperature: Override default temperature
            top_p: Override default top_p

        Returns:
            Generated response text
        """
        formatted_prompt = self._format_prompt(prompt, system_prompt)

        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Override generation config if needed
        gen_config = self.generation_config
        if any(x is not None for x in [max_new_tokens, temperature, top_p]):
            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                temperature=temperature or self.temperature,
                top_p=top_p or self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config
            )

        # Decode output, skipping the input prompt
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's response
        if "<|assistant|>" in full_output:
            response = full_output.split("<|assistant|>")[-1].strip()
        else:
            response = full_output[len(formatted_prompt):].strip()

        return response

    def generate_multi_turn(
        self,
        turns: List[str],
        system_prompt: Optional[str] = None,
        return_all_responses: bool = True
    ) -> List[str]:
        """
        Generate responses for multi-turn conversation

        Args:
            turns: List of user prompts in sequence
            system_prompt: Optional system instruction
            return_all_responses: If True, return all responses; if False, only final

        Returns:
            List of assistant responses (or single final response if return_all_responses=False)
        """
        responses = []
        conversation_history = []

        for i, turn in enumerate(turns):
            # Build conversation context
            if i == 0:
                # First turn
                formatted_prompt = self._format_prompt(turn, system_prompt)
            else:
                # Subsequent turns - include history
                formatted_prompt = self._format_prompt(turn, system_prompt)
                # Add previous exchanges
                for prev_turn, prev_response in zip(turns[:i], responses):
                    formatted_prompt = (
                        f"<|user|>\n{prev_turn}</s>\n"
                        f"<|assistant|>\n{prev_response}</s>\n"
                        f"{formatted_prompt}"
                    )

            # Generate response
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config
                )

            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract response
            if "<|assistant|>" in full_output:
                response = full_output.split("<|assistant|>")[-1].strip()
            else:
                response = full_output[len(formatted_prompt):].strip()

            responses.append(response)

        return responses if return_all_responses else [responses[-1]]

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        batch_size: int = 4
    ) -> List[str]:
        """
        Generate responses for multiple prompts in batches

        Args:
            prompts: List of user prompts
            system_prompt: Optional system instruction
            batch_size: Number of prompts to process simultaneously

        Returns:
            List of generated responses
        """
        all_responses = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            formatted_batch = [self._format_prompt(p, system_prompt) for p in batch]

            # Tokenize batch
            inputs = self.tokenizer(
                formatted_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config
                )

            # Decode each output
            for output in outputs:
                full_output = self.tokenizer.decode(output, skip_special_tokens=True)
                if "<|assistant|>" in full_output:
                    response = full_output.split("<|assistant|>")[-1].strip()
                else:
                    # Fallback: just remove the formatted prompt prefix
                    response = full_output.strip()
                all_responses.append(response)

        return all_responses

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample
        }

    def __repr__(self):
        return f"TinyLlamaIntegration(model={self.model_name}, device={self.device})"
