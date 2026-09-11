import pytest
from app.services.prompt_compressor import PromptCompressor


def test_whitespace_compression():
    text = "Line 1\n\n\n\nLine 2     with    spaces"
    cleaned = PromptCompressor.clean_whitespace(text)
    assert cleaned == "Line 1\n\nLine 2 with spaces"


def test_filler_pruning():
    verbose = (
        "Hello dear AI assistant! Could you please be so kind as to "
        "explain what is Docker? Thank you very much in advance!"
    )
    compressed, orig_tok, comp_tok, saved_tok = PromptCompressor.compress(verbose, aggressiveness="moderate")

    assert "Docker" in compressed
    assert "Hello dear AI assistant" not in compressed
    assert "Thank you very much in advance" not in compressed
    assert comp_tok < orig_tok
    assert saved_tok > 0


def test_empty_string_compression():
    compressed, orig_tok, comp_tok, saved = PromptCompressor.compress("")
    assert compressed == ""
    assert orig_tok == 0
    assert comp_tok == 0
    assert saved == 0
