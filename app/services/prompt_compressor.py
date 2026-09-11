import re
from typing import Tuple, List
from app.core.telemetry import estimate_tokens
from app.core.config import settings

# Common conversational filler and verbose boilerplate regex patterns
FILLER_PATTERNS = [
    r"(?i)\b(hello|hi|hey|greetings|good morning|good afternoon|good evening)\b[,!.]*",
    r"(?i)\b(i hope this message finds you well|i hope you are having a wonderful day|hope you are doing well)\b[,!.]*",
    r"(?i)\b(could you please be so kind as to|could you please|can you please|would you kindly|kindly)\b",
    r"(?i)\b(do me a small favor and|do me a favor and)\b",
    r"(?i)\b(i would truly appreciate it if you could|i would appreciate your help|i would appreciate it if you)\b",
    r"(?i)\b(thank you very much in advance|thanks in advance|thank you so much|thank you)\b[!.]*",
    r"(?i)\b(as an ai assistant|as an expert|as a helpful assistant)\b[,!.]*",
    r"(?i)\b(please feel free to|in this request i am writing to)\b",
]

# Secondary filler words removed in aggressive mode
AGGRESSIVE_FILLERS = [
    r"(?i)\b(basically|literally|actually|honestly|obviously|clearly)\b",
    r"(?i)\b(in order to)\b",  # -> "to"
    r"(?i)\b(at this point in time)\b",  # -> "now"
]


class PromptCompressor:
    """
    Optimizes and compresses prompts by stripping whitespace bloat,
    conversational pleasantries, boilerplate, and low-information tokens.
    """

    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Normalizes multiple newlines and spaces."""
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    @classmethod
    def compress(
        cls,
        text: str,
        aggressiveness: str = "moderate"
    ) -> Tuple[str, int, int, int]:
        """
        Compresses input prompt according to aggressiveness level.
        Returns: (compressed_text, original_tokens, compressed_tokens, tokens_saved)
        """
        if not text:
            return "", 0, 0, 0

        orig_tokens = estimate_tokens(text)
        processed = cls.clean_whitespace(text)

        removed_filler_count = 0

        if aggressiveness in ("moderate", "aggressive"):
            for pattern in FILLER_PATTERNS:
                matches = len(re.findall(pattern, processed))
                if matches > 0:
                    removed_filler_count += matches
                    processed = re.sub(pattern, "", processed)

        if aggressiveness == "aggressive":
            # Replace verbose phrases with concise equivalents
            processed = re.sub(r"(?i)\bin order to\b", "to", processed)
            processed = re.sub(r"(?i)\bat this point in time\b", "now", processed)
            for pattern in AGGRESSIVE_FILLERS:
                processed = re.sub(pattern, "", processed)

        # Final cleanup of leftover double punctuation or spaces
        processed = re.sub(r"\s+([,.?!])", r"\1", processed)
        processed = re.sub(r"([,.?!]){2,}", r"\1", processed)
        processed = cls.clean_whitespace(processed)

        # Fallback: if over-pruning emptied the text, restore cleaned original
        if not processed:
            processed = cls.clean_whitespace(text)

        # Enforce max length if set
        if len(processed) > settings.compression_max_length:
            processed = processed[:settings.compression_max_length]

        compressed_tokens = estimate_tokens(processed)
        tokens_saved = max(0, orig_tokens - compressed_tokens)

        return processed, orig_tokens, compressed_tokens, tokens_saved


prompt_compressor = PromptCompressor()
