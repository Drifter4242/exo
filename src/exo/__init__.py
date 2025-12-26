# type: ignore
# Monkey-patch for transformers 5.0 compatibility
# The bytes_to_unicode function was removed in transformers 5.0, but some
# HuggingFace model tokenizers (like Kimi-K2) still import it.
# We inject it back to maintain compatibility.


def _bytes_to_unicode():
    """
    Returns a mapping from bytes to unicode characters.
    This was removed from transformers.models.gpt2.tokenization_gpt2 in v5.0.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs, strict=False))


def _fix_tokenizer_special_tokens_pattern(tokenizer):
    """
    Fix tokenizer special_tokens_pattern for transformers 5.0 compatibility.

    Transformers 5.0 introduced a new special_tokens_pattern system that defaults
    to 'cls_sep'. Tokenizers like Kimi-K2 use bos/eos tokens instead, causing
    None values to be inserted when cls_token_id/sep_token_id are not set.
    """
    # If pattern expects cls/sep but tokenizer has bos/eos instead
    if (
        hasattr(tokenizer, "special_tokens_pattern")
        and tokenizer.special_tokens_pattern == "cls_sep"
        and tokenizer.cls_token_id is None
        and tokenizer.bos_token_id is not None
    ):
        tokenizer.special_tokens_pattern = "bos_eos"
    return tokenizer


try:
    import transformers.models.gpt2.tokenization_gpt2 as _gpt2_tok

    if not hasattr(_gpt2_tok, "bytes_to_unicode"):
        _gpt2_tok.bytes_to_unicode = _bytes_to_unicode

    # Wrap AutoTokenizer.from_pretrained to fix special_tokens_pattern
    from transformers import AutoTokenizer as _AutoTokenizer

    _original_from_pretrained = _AutoTokenizer.from_pretrained.__func__

    @classmethod
    def _patched_from_pretrained(cls, *args, **kwargs):
        tokenizer = _original_from_pretrained(cls, *args, **kwargs)
        return _fix_tokenizer_special_tokens_pattern(tokenizer)

    _AutoTokenizer.from_pretrained = _patched_from_pretrained
except ImportError:
    pass  # transformers not installed
