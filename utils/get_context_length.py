from transformers import AutoConfig
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_context_length(model_name: str) -> int:
    """Get maximum context length from model config."""
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Check various possible context length attributes
        context_length = (
            getattr(config, 'max_position_embeddings', None) or
            getattr(config, 'sliding_window', None) or
            getattr(config, 'max_sequence_length', None) or
            getattr(config, 'max_seq_len', None) or
            4096  # Default fallback
        )

        # Some models (like Qwen) might have sliding_window disabled
        if hasattr(config, 'use_sliding_window') and not config.use_sliding_window:
            # If sliding window is disabled, use max_position_embeddings instead
            context_length = getattr(config, 'max_position_embeddings', context_length)
            

        # Cap to 32k
        return min(context_length, 32768)
    except Exception as e:
        logger.warning(f"Could not get context length from config for {model_name}: {e}")
        return 4096  # Default fallback


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    print(get_context_length(args.model_name))
