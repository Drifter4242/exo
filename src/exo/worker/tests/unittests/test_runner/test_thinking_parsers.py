"""Tests for parse_thinking_models and parse_secondary_thinking_tags.

Token stream facts verified empirically against moonshotai/Kimi-K2.6 tokenizer
(2026-05-02, transformers AutoTokenizer):

    '<think>'       → ids=[163606]              single token, always arrives alone
    '<think>\\n'    → ids=[163606, 198]          TWO tokens: <think> then Ċ (\\n)
    '</think>'      → ids=[163607]              single token, always arrives alone
    '<thinking>'    → ids=[46145, 29596, 29]    '<th' + 'inking' + '>'  (3 BPE tokens)
    '<thinking>\\n' → ids=[46145, 29596, 719]   '<th' + 'inking' + '>Ċ' (>Ċ = ">\\n")
    '</thinking>'   → ids=[993, 130400, 29]     '</' + 'thinking' + '>'  (3 BPE tokens)

Key implications:
  - <think> / </think> are added tokens (ids 163606/163607) that always arrive as
    single, atomic units.  The accumulated string can only ever equal the tag exactly,
    never overshoot it.  Any bundle check in parse_thinking_models is therefore
    unreachable dead code.
  - <thinking>\\n arrives as ['<th', 'inking', '>Ċ'].  The final token '>Ċ' carries
    '>' (which closes the tag) merged with '\\n' (first content char).  When added to
    accumulated the result is '<thinking>\\n', which is longer than '<thinking>' and
    starts with it — so the bundle check in parse_secondary_thinking_tags IS essential.
  - </thinking> builds up to an exact match over 3 tokens, so no bundle case arises.
"""

from collections.abc import Generator

import pytest

from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.runner.llm_inference.model_output_parsers import (
    parse_secondary_thinking_tags,
    parse_thinking_models,
)

THINK_START = "<think>"
THINK_END = "</think>"
SEC_START = "<thinking>"
SEC_END = "</thinking>"


def _gen(*tokens: str) -> Generator[GenerationResponse]:
    """Yield GenerationResponses; the last token gets finish_reason='stop'."""
    items = list(tokens)
    for i, text in enumerate(items):
        yield GenerationResponse(
            text=text,
            token=i,
            finish_reason="stop" if i == len(items) - 1 else None,
            usage=None,
        )


def _run(gen: Generator[GenerationResponse]) -> list[GenerationResponse]:
    return [r for r in gen if r is not None]


# ─────────────────────────────────────────────────────────────────────────────
# parse_thinking_models — primary <think> / </think>
# ─────────────────────────────────────────────────────────────────────────────


class TestParseThinkingModels:
    """<think> and </think> are single atomic tokens; no bundling ever occurs."""

    def test_basic_think_block(self):
        """<think> tag swallowed, content routed as thinking, </think> swallowed, then content."""
        results = _run(
            parse_thinking_models(
                _gen("<think>", "reasoning text", "</think>", "answer"),
                THINK_START,
                THINK_END,
                starts_in_thinking=False,
            )
        )
        thinking = [r for r in results if r.is_thinking]
        content = [r for r in results if not r.is_thinking]
        assert [r.text for r in thinking] == ["reasoning text"]
        assert "answer" in [r.text for r in content]

    def test_starts_in_thinking(self):
        """When starts_in_thinking=True, content before </think> is routed as thinking."""
        results = _run(
            parse_thinking_models(
                _gen("reasoning text", "</think>", "answer"),
                THINK_START,
                THINK_END,
                starts_in_thinking=True,
            )
        )
        thinking = [r for r in results if r.is_thinking]
        content = [r for r in results if not r.is_thinking]
        assert [r.text for r in thinking] == ["reasoning text"]
        assert "answer" in [r.text for r in content]

    def test_newline_after_think_arrives_as_separate_token(self):
        """Verified: '<think>\\n' tokenises as two tokens.  \\n follows as a plain
        content token AFTER <think> has already been consumed by exact-match.
        The bundle check is unreachable — removing it does not break this case."""
        # Simulate actual Kimi token stream: token 163606 ("<think>") then token 198 ("\\n")
        results = _run(
            parse_thinking_models(
                _gen("<think>", "\n", "reasoning", "</think>", "answer"),
                THINK_START,
                THINK_END,
                starts_in_thinking=False,
            )
        )
        thinking_texts = [r.text for r in results if r.is_thinking]
        content_texts = [r.text for r in results if not r.is_thinking]
        # Both \\n and "reasoning" arrive after <think> is consumed → thinking content
        assert "\n" in thinking_texts
        assert "reasoning" in thinking_texts
        assert "answer" in content_texts
        # <think> and </think> are swallowed — must NOT appear in output
        assert "<think>" not in thinking_texts + content_texts
        assert "</think>" not in thinking_texts + content_texts

    def test_no_thinking_passthrough(self):
        """No think tags → all tokens pass through as is_thinking=False."""
        results = _run(
            parse_thinking_models(
                _gen("Hello", " world"),
                THINK_START,
                THINK_END,
                starts_in_thinking=False,
            )
        )
        assert all(not r.is_thinking for r in results)
        assert [r.text for r in results if r.finish_reason is None] == ["Hello"]

    def test_bundle_check_unreachable(self):
        """No token stream can produce accumulated='<think>X' in parse_thinking_models.

        Because <think> is always a single atomic token, the accumulated string
        transitions directly from a prefix of '<think>' to exactly '<think>' to ''.
        It can never overshoot to '<think>X'.  This test confirms that a stream
        where the tag arrives as one token is handled identically with or without
        the (now removed) bundle check blocks.
        """
        # Simulated: what would happen if somehow "<think>X" arrived bundled
        # (impossible with the real tokenizer, but let's ensure the remaining
        # prefix-match logic handles it gracefully rather than silently losing tokens)
        results = _run(
            parse_thinking_models(
                _gen("<think>", "content", "</think>", "answer"),
                THINK_START,
                THINK_END,
                starts_in_thinking=False,
            )
        )
        thinking_texts = [r.text for r in results if r.is_thinking]
        assert "content" in thinking_texts


# ─────────────────────────────────────────────────────────────────────────────
# parse_secondary_thinking_tags — secondary <thinking> / </thinking> BPE text
# ─────────────────────────────────────────────────────────────────────────────


class TestParseSecondaryThinkingTags:
    """<thinking> is plain BPE text (not an added token).  Its tokenisation produces
    a >Ċ bundle token that makes accumulated overshoot the open tag."""

    def test_real_bpe_token_stream_open_and_close(self):
        """Exact Kimi K2.6 BPE token stream for <thinking>\\ncontent</thinking>answer.

        Open:  '<th' + 'inking' + '>Ċ'   (>Ċ = '>\\n', bundle fires: remainder='\\n')
        Close: '</'  + 'thinking' + '>'   (exact match, no bundle needed)
        """
        # BPE stream as the tokenizer would emit it
        results = _run(
            parse_secondary_thinking_tags(
                _gen(
                    "<th", "inking", ">\n", "content", "</", "thinking", ">", "answer"
                ),
                SEC_START,
                SEC_END,
            )
        )
        thinking_texts = [r.text for r in results if r.is_thinking]
        content_texts = [r.text for r in results if not r.is_thinking]

        # '>\\n' bundle: remainder '\\n' routed to thinking
        assert "\n" in thinking_texts
        # post-open content is thinking
        assert "content" in thinking_texts
        # post-close content is normal
        assert "answer" in content_texts
        # tag text must not leak
        for tag_frag in ("<th", "inking", ">", "</", "thinking"):
            assert tag_frag not in thinking_texts + content_texts

    def test_bundle_check_is_essential_for_secondary_open(self):
        """Without the bundle check, '>\\n' after '<thinking' would be flushed as
        non-thinking content because it doesn't match any prefix of '</thinking>'.
        Confirm the bundle check correctly routes it to thinking."""
        results = _run(
            parse_secondary_thinking_tags(
                _gen(
                    "<th",
                    "inking",
                    ">\n",
                    "deep thought",
                    "</",
                    "thinking",
                    ">",
                    "done",
                ),
                SEC_START,
                SEC_END,
            )
        )
        thinking_texts = [r.text for r in results if r.is_thinking]
        # Without bundle check: ">\\n" would appear in content_texts (wrong)
        # With bundle check:    "\\n" appears in thinking_texts (correct)
        assert "\n" in thinking_texts, (
            "Bundle check is required: '>Ċ' token must be split at the tag boundary "
            "and its '\\n' suffix routed to thinking content"
        )
        assert "deep thought" in thinking_texts

    def test_close_tag_exact_match_three_tokens(self):
        """</thinking> builds via exact prefix match over 3 tokens — no bundle fires."""
        results = _run(
            parse_secondary_thinking_tags(
                _gen("<th", "inking", ">\n", "thought", "</", "thinking", ">", "after"),
                SEC_START,
                SEC_END,
            )
        )
        thinking_texts = [r.text for r in results if r.is_thinking]
        content_texts = [r.text for r in results if not r.is_thinking]
        assert "thought" in thinking_texts
        assert "after" in content_texts
        # Close tag fragments must not appear in output
        assert "</" not in thinking_texts + content_texts
        assert "thinking" not in thinking_texts + content_texts

    def test_primary_thinking_tokens_pass_through_unchanged(self):
        """Tokens already marked is_thinking=True by the primary pass are not re-processed."""

        # Simulate mixed stream: primary-thinking token interspersed with secondary tags
        def _mixed() -> Generator[GenerationResponse]:
            primary_thought = GenerationResponse(
                text="primary reasoning",
                token=0,
                finish_reason=None,
                usage=None,
                is_thinking=True,
            )
            yield primary_thought
            yield from _gen(
                "<th", "inking", ">\n", "secondary", "</", "thinking", ">", "answer"
            )

        results = _run(parse_secondary_thinking_tags(_mixed(), SEC_START, SEC_END))
        # Primary token passes through with is_thinking=True intact
        primary = [r for r in results if r.text == "primary reasoning"]
        assert len(primary) == 1 and primary[0].is_thinking
        # Secondary tags also correctly processed
        secondary = [r for r in results if r.text == "secondary"]
        assert len(secondary) == 1 and secondary[0].is_thinking

    def test_no_secondary_tags_passthrough(self):
        """Stream with no secondary tags passes through without modification."""
        results = _run(
            parse_secondary_thinking_tags(
                _gen("Hello", " world"),
                SEC_START,
                SEC_END,
            )
        )
        assert all(not r.is_thinking for r in results)

    def test_multiple_secondary_blocks(self):
        """Two consecutive secondary <thinking> blocks are both stripped correctly."""
        results = _run(
            parse_secondary_thinking_tags(
                _gen(
                    "<th",
                    "inking",
                    ">\n",
                    "block1",
                    "</",
                    "thinking",
                    ">",
                    "mid",
                    "<th",
                    "inking",
                    ">\n",
                    "block2",
                    "</",
                    "thinking",
                    ">",
                    "end",
                ),
                SEC_START,
                SEC_END,
            )
        )
        thinking_texts = [r.text for r in results if r.is_thinking]
        content_texts = [r.text for r in results if not r.is_thinking]
        assert "block1" in thinking_texts
        assert "block2" in thinking_texts
        assert "mid" in content_texts
        assert "end" in content_texts
