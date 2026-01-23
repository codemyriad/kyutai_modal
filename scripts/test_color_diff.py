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
    - "dim": text missing from actual (was in expected)

    Uses longest common subsequence to handle insertions/deletions.
    """
    if not actual:
        return [("...", "dim")]

    exp_words = expected.split()
    act_words = actual.split()

    # Build LCS table
    m, n = len(exp_words), len(act_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words_match(exp_words[i-1], act_words[j-1]):
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Backtrack to find alignment
    segments = []
    i, j = m, n

    # Collect operations in reverse
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and words_match(exp_words[i-1], act_words[j-1]):
            ops.append(("match", act_words[j-1]))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] >= dp[i-1][j]):
            ops.append(("extra", act_words[j-1]))
            j -= 1
        else:
            ops.append(("missing", exp_words[i-1]))
            i -= 1

    ops.reverse()

    # Find where trailing missing words start (future text, not errors)
    trailing_start = len(ops)
    for idx in range(len(ops) - 1, -1, -1):
        if ops[idx][0] == "missing":
            trailing_start = idx
        else:
            break

    # Convert ops to segments, merging consecutive same-color
    for idx, (op, word) in enumerate(ops):
        if op == "match":
            color = "green"
        elif op == "extra":
            color = "red"
        elif idx >= trailing_start:
            color = "pending"  # future text, just dim
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
