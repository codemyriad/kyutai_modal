"""Tests for color_diff segmentation."""

import pytest


def normalize_word(w: str) -> str:
    """Normalize a word for comparison."""
    return w.lower().strip(",.!?;:'\"")


def words_match(w1: str, w2: str) -> bool:
    """Check if two words match, allowing for minor spelling differences."""
    n1 = normalize_word(w1)
    n2 = normalize_word(w2)
    if n1 == n2:
        return True
    # Allow edit distance of 1 for words of 5+ chars (handles memorising/memorizing)
    if len(n1) >= 5 and len(n2) >= 5:
        if abs(len(n1) - len(n2)) <= 1:
            # Simple check: differ by at most 1 char
            diffs = sum(1 for a, b in zip(n1, n2) if a != b)
            diffs += abs(len(n1) - len(n2))
            return diffs <= 1
    return False


def color_diff(expected: str, actual: str) -> list[tuple[str, str]]:
    """
    Compare expected and actual text, return segments with colors.

    Returns list of (text, color) tuples where color is:
    - "green": matching text
    - "red": text in actual that differs from expected
    - "dim": text missing from actual (was in expected) - strikethrough
    - "pending": text not yet reached in actual - just dim

    Uses greedy prefix-aligned matching optimized for streaming transcription.
    Each actual word matches with the earliest available expected word.
    """
    if not actual:
        return [("...", "dim")]

    exp_words = expected.split()
    act_words = actual.split()

    # Greedy prefix-aligned matching
    ops = []
    exp_idx = 0

    for act_word in act_words:
        # Find the first matching expected word from current position
        found_idx = None
        for i in range(exp_idx, len(exp_words)):
            if words_match(exp_words[i], act_word):
                found_idx = i
                break

        if found_idx is not None:
            # Mark all skipped expected words as missing
            for j in range(exp_idx, found_idx):
                ops.append(("missing", exp_words[j]))
            # Mark this as a match
            ops.append(("match", act_word))
            exp_idx = found_idx + 1
        else:
            # No match found - this is an extra word
            ops.append(("extra", act_word))

    # Remaining expected words are pending (trailing/future text)
    for i in range(exp_idx, len(exp_words)):
        ops.append(("pending", exp_words[i]))

    # Convert ops to segments, merging consecutive same-color
    segments = []
    for op, word in ops:
        if op == "match":
            color = "green"
        elif op == "extra":
            color = "red"
        elif op == "pending":
            color = "pending"
        else:
            color = "dim"  # incorrectly skipped, strikethrough

        if segments and segments[-1][1] == color:
            segments[-1] = (segments[-1][0] + " " + word, color)
        else:
            segments.append((word, color))

    return segments


def render_colored(segments: list[tuple[str, str]]) -> str:
    """Render segments as Rich markup."""
    parts = []
    for text, color in segments:
        if color == "dim":
            parts.append(f"[dim strike]{text}[/dim strike]")
        elif color == "pending":
            parts.append(f"[dim]{text}[/dim]")
        else:
            parts.append(f"[{color}]{text}[/{color}]")
    return " ".join(parts)


# Test cases
class TestColorDiff:
    def test_exact_match(self):
        """Identical text should be all green."""
        expected = "hello world"
        actual = "hello world"
        segments = color_diff(expected, actual)
        assert segments == [("hello world", "green")]

    def test_missing_start(self):
        """Missing words at start should show actual as green, not red."""
        expected = "No, I mean, why are they"
        actual = "I mean, why are they"
        segments = color_diff(expected, actual)
        # "No," is missing (dim), rest matches (green)
        assert ("No,", "dim") in segments
        assert any(s[1] == "green" and "I mean" in s[0] for s in segments)

    def test_missing_middle(self):
        """Missing words in middle."""
        expected = "the quick brown fox jumps"
        actual = "the quick fox jumps"
        segments = color_diff(expected, actual)
        # "brown" should be dim (missing)
        assert any("brown" in s[0] and s[1] == "dim" for s in segments)

    def test_extra_words(self):
        """Extra words in actual should be red."""
        expected = "hello world"
        actual = "hello beautiful world"
        segments = color_diff(expected, actual)
        # "beautiful" is extra (red)
        assert any("beautiful" in s[0] and s[1] == "red" for s in segments)

    def test_spelling_variation(self):
        """Spelling variations like memorising/memorizing should match."""
        expected = "memorising that"
        actual = "memorizing that"
        segments = color_diff(expected, actual)
        # Both words should be green (fuzzy match)
        assert segments == [("memorizing that", "green")]

    def test_empty_actual(self):
        """Empty actual returns dim placeholder."""
        segments = color_diff("hello world", "")
        assert segments == [("...", "dim")]

    def test_case_insensitive(self):
        """Matching should be case insensitive."""
        expected = "Hello World"
        actual = "hello world"
        segments = color_diff(expected, actual)
        assert segments == [("hello world", "green")]

    def test_punctuation_ignored(self):
        """Punctuation differences shouldn't cause mismatch."""
        expected = "Hello, world!"
        actual = "Hello world"
        segments = color_diff(expected, actual)
        assert segments == [("Hello world", "green")]

    def test_real_example(self):
        """Real example from the user."""
        expected = "No, I mean, why are they still sometimes so brittle, memorising that Tom Smith's wife is Mary Stone"
        actual = "I mean, why are they still sometimes so brittle, memorizing that Tom Smith's wife is Mary Stone"
        segments = color_diff(expected, actual)

        # "No," should be missing (dim)
        assert any("No," in s[0] and s[1] == "dim" for s in segments)

        # Most of the rest should be green (matching)
        green_text = " ".join(s[0] for s in segments if s[1] == "green")
        assert "I mean" in green_text
        assert "Mary Stone" in green_text

    def test_partial_transcription(self):
        """Partial transcription should show missing tail as pending (not strikethrough)."""
        expected = (
            "For an LLM, the first time they hear, Tom Smith's wife is Mary, "
            "that just updates their weights as to predicting what comes after, "
            "in the future, Tom Smith's wife is, or maybe permutations like, "
            "the wife of Tom Smith is."
        )
        actual = "For an LLM, the first time they hear, Tom Smith's wife is Mary, that just updates"
        segments = color_diff(expected, actual)

        # The matching prefix should be green
        green_text = " ".join(s[0] for s in segments if s[1] == "green")
        assert "For an LLM" in green_text
        assert "Tom Smith's wife is Mary" in green_text
        assert "that just updates" in green_text

        # The missing tail should be pending (future text, no strikethrough)
        pending_text = " ".join(s[0] for s in segments if s[1] == "pending")
        assert "their weights" in pending_text
        assert "predicting what comes after" in pending_text
        assert "the wife of Tom Smith is." in pending_text

        # No red (no extra words in actual) and no dim (no incorrectly skipped words)
        assert not any(s[1] == "red" for s in segments)
        assert not any(s[1] == "dim" for s in segments)


class TestProgressiveRecognition:
    """Test color_diff at various stages of recognition completion."""

    EXPECTED = (
        "No, I mean, why are they still sometimes so brittle, "
        "memorising that Tom Smith's wife is Mary Stone, "
        "but not deducing that Mary Stone's husband is Tom Smith?"
    )

    def test_stage_1_very_early(self):
        """Very early recognition - just first few words."""
        actual = "No, I mean"
        segments = color_diff(self.EXPECTED, actual)

        # "No, I mean" should be green
        green_text = " ".join(s[0] for s in segments if s[1] == "green")
        assert "No," in green_text
        assert "I mean" in green_text or "mean" in green_text

        # Rest should be pending (not strikethrough)
        pending_text = " ".join(s[0] for s in segments if s[1] == "pending")
        assert "why are they" in pending_text
        assert "Tom Smith" in pending_text

        # No incorrectly skipped words
        assert not any(s[1] == "dim" for s in segments)
        assert not any(s[1] == "red" for s in segments)

    def test_stage_2_early(self):
        """Early recognition - partial first clause."""
        actual = "No, I mean, why are they still sometimes"
        segments = color_diff(self.EXPECTED, actual)

        green_text = " ".join(s[0] for s in segments if s[1] == "green")
        assert "No," in green_text
        assert "why are they" in green_text
        assert "sometimes" in green_text

        pending_text = " ".join(s[0] for s in segments if s[1] == "pending")
        assert "so brittle" in pending_text
        assert "memorising" in pending_text

        assert not any(s[1] == "dim" for s in segments)
        assert not any(s[1] == "red" for s in segments)

    def test_stage_3_mid_sentence(self):
        """Mid-sentence - through first name mention.

        Greedy algorithm matches "Mary" with the first occurrence in expected,
        so all remaining text is correctly marked as pending.
        """
        actual = "No, I mean, why are they still sometimes so brittle, memorizing that Tom Smith's wife is Mary"
        segments = color_diff(self.EXPECTED, actual)

        green_text = " ".join(s[0] for s in segments if s[1] == "green")
        assert "No," in green_text
        assert "brittle" in green_text
        assert "Tom Smith's" in green_text
        assert "Mary" in green_text

        # "memorising" vs "memorizing" should fuzzy match (green)
        assert "memorizing" in green_text or "memorising" in green_text

        # All remaining text should be pending (not strikethrough)
        pending_text = " ".join(s[0] for s in segments if s[1] == "pending")
        assert "Stone," in pending_text
        assert "but not deducing" in pending_text
        assert "Tom Smith?" in pending_text

        # No incorrectly skipped words
        assert not any(s[1] == "dim" for s in segments)
        assert not any(s[1] == "red" for s in segments)

    def test_stage_4_near_complete(self):
        """Near complete - missing just the end."""
        actual = "No, I mean, why are they still sometimes so brittle, memorizing that Tom Smith's wife is Mary Stone, but not deducing that Mary Stone's husband is"
        segments = color_diff(self.EXPECTED, actual)

        green_text = " ".join(s[0] for s in segments if s[1] == "green")
        assert "No," in green_text
        assert "Mary Stone," in green_text or "Mary Stone" in green_text
        assert "but not deducing" in green_text
        assert "husband is" in green_text

        pending_text = " ".join(s[0] for s in segments if s[1] == "pending")
        assert "Tom Smith?" in pending_text or "Tom Smith" in pending_text

        assert not any(s[1] == "dim" for s in segments)
        assert not any(s[1] == "red" for s in segments)

    def test_stage_5_complete_exact(self):
        """Complete recognition - exact match (with spelling variation)."""
        actual = "No, I mean, why are they still sometimes so brittle, memorizing that Tom Smith's wife is Mary Stone, but not deducing that Mary Stone's husband is Tom Smith?"
        segments = color_diff(self.EXPECTED, actual)

        # Everything should be green (fuzzy match handles memorising/memorizing)
        assert all(s[1] == "green" for s in segments)

        green_text = " ".join(s[0] for s in segments if s[1] == "green")
        assert "Tom Smith?" in green_text

    def test_stage_6_missing_start(self):
        """Complete but missing "No," at start - should be strikethrough."""
        actual = "I mean, why are they still sometimes so brittle, memorizing that Tom Smith's wife is Mary Stone, but not deducing that Mary Stone's husband is Tom Smith?"
        segments = color_diff(self.EXPECTED, actual)

        # "No," should be dim (strikethrough - incorrectly skipped)
        dim_text = " ".join(s[0] for s in segments if s[1] == "dim")
        assert "No," in dim_text

        # Rest should be green
        green_text = " ".join(s[0] for s in segments if s[1] == "green")
        assert "I mean," in green_text
        assert "Tom Smith?" in green_text

        # No pending (nothing at the end missing)
        assert not any(s[1] == "pending" for s in segments)

    def test_stage_7_missing_middle(self):
        """Missing words in the middle - should be strikethrough."""
        actual = "No, I mean, why are they still sometimes so brittle, memorizing that Tom Smith's wife is Mary Stone, but not deducing that husband is Tom Smith?"
        # Missing "Mary Stone's" before "husband"
        segments = color_diff(self.EXPECTED, actual)

        # "Mary Stone's" should be dim (strikethrough)
        dim_text = " ".join(s[0] for s in segments if s[1] == "dim")
        assert "Mary" in dim_text and "Stone's" in dim_text

        # Start and end should be green
        green_text = " ".join(s[0] for s in segments if s[1] == "green")
        assert "No," in green_text
        assert "Tom Smith?" in green_text

    def test_stage_8_extra_words(self):
        """Extra words inserted - should be red."""
        actual = "No, I mean, why are they still sometimes so very brittle, memorizing that Tom Smith's wife is Mary Stone, but not deducing that Mary Stone's husband is Tom Smith?"
        # Extra "very" before "brittle"
        segments = color_diff(self.EXPECTED, actual)

        # "very" should be red (extra word)
        red_text = " ".join(s[0] for s in segments if s[1] == "red")
        assert "very" in red_text

        # Rest should be green
        green_text = " ".join(s[0] for s in segments if s[1] == "green")
        assert "No," in green_text
        assert "so" in green_text
        assert "brittle," in green_text or "brittle" in green_text

    def test_stage_9_wrong_word(self):
        """Wrong word substituted - shows as missing + extra."""
        actual = "No, I mean, why are they still sometimes so fragile, memorizing that Tom Smith's wife is Mary Stone, but not deducing that Mary Stone's husband is Tom Smith?"
        # "fragile" instead of "brittle"
        segments = color_diff(self.EXPECTED, actual)

        # "brittle," should be dim (missing/strikethrough)
        dim_text = " ".join(s[0] for s in segments if s[1] == "dim")
        assert "brittle" in dim_text

        # "fragile" should be red (extra)
        red_text = " ".join(s[0] for s in segments if s[1] == "red")
        assert "fragile" in red_text

        # Surrounding text should be green
        green_text = " ".join(s[0] for s in segments if s[1] == "green")
        assert "so" in green_text
        assert "memorizing" in green_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
