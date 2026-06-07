"""Format-specific text extraction."""

from __future__ import annotations

from recall.connectors.text_extraction import (
    extract_html,
    extract_json,
    extract_markdown,
    extract_text,
    normalize_whitespace,
)


class TestNormalizeWhitespace:
    def test_collapses_horizontal_runs(self) -> None:
        assert normalize_whitespace("a  \t b") == "a b"

    def test_caps_blank_lines_at_one(self) -> None:
        assert normalize_whitespace("a\n\n\n\n\nb") == "a\n\nb"

    def test_normalizes_line_endings(self) -> None:
        assert normalize_whitespace("a\r\nb\rc") == "a\nb\nc"

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        assert normalize_whitespace("\n\n  hello  \n\n") == "hello"


class TestHtml:
    def test_extracts_the_title(self) -> None:
        _, title = extract_html("<html><head><title>Guide</title></head><body>x</body></html>")
        assert title == "Guide"

    def test_drops_script_and_style_content(self) -> None:
        html = (
            "<html><head><style>body{color:red}</style></head>"
            "<body><p>visible</p><script>secret()</script></body></html>"
        )
        text, _ = extract_html(html)
        assert "visible" in text
        assert "secret" not in text
        assert "color:red" not in text

    def test_block_tags_become_line_breaks(self) -> None:
        text, _ = extract_html("<p>one</p><p>two</p>")
        assert text.splitlines() == ["one", "two"]

    def test_decodes_character_references(self) -> None:
        text, _ = extract_html("<p>a &amp; b &lt;c&gt;</p>")
        assert text == "a & b <c>"

    def test_missing_title_returns_none(self) -> None:
        _, title = extract_html("<p>no title here</p>")
        assert title is None


class TestJson:
    def test_flattens_nested_structures(self) -> None:
        text, _ = extract_json('{"a": {"b": 1}, "c": [10, 20]}')
        assert "a.b: 1" in text
        assert "c[0]: 10" in text
        assert "c[1]: 20" in text

    def test_lifts_a_title_like_key(self) -> None:
        _, title = extract_json('{"name": "billing-service"}')
        assert title == "billing-service"

    def test_malformed_json_falls_back_to_raw_text(self) -> None:
        text, title = extract_json("{not json at all")
        assert "not json at all" in text
        assert title is None

    def test_null_values_are_skipped(self) -> None:
        text, _ = extract_json('{"a": null, "b": 2}')
        assert "a:" not in text
        assert "b: 2" in text


class TestMarkdown:
    def test_lifts_the_first_heading(self) -> None:
        _, title = extract_markdown("# Authentication\n\nbody text")
        assert title == "Authentication"

    def test_ignores_headings_after_body_text(self) -> None:
        _, title = extract_markdown("intro paragraph\n\n# Later Heading")
        assert title is None

    def test_preserves_markdown_syntax(self) -> None:
        text, _ = extract_markdown("# H\n\n- item one\n- item two")
        assert "- item one" in text


class TestPlainText:
    def test_passes_through_with_normalisation(self) -> None:
        text, title = extract_text("  hello   world  ")
        assert text == "hello world"
        assert title is None
