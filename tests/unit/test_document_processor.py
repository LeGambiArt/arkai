"""Unit tests for document processor."""

import tempfile
from pathlib import Path

from arkai import document_processor


class TestFormatDetection:
    def test_detect_txt_format(self):
        """Test detection of text format."""
        assert document_processor.detect_format("document.txt") == "txt"

    def test_detect_markdown_format(self):
        """Test detection of markdown format."""
        assert document_processor.detect_format("readme.md") == "markdown"

    def test_detect_html_format(self):
        """Test detection of HTML format."""
        assert document_processor.detect_format("page.html") == "html"
        assert document_processor.detect_format("page.htm") == "html"

    def test_detect_pdf_format(self):
        """Test detection of PDF format."""
        assert document_processor.detect_format("document.pdf") == "pdf"

    def test_detect_latex_format(self):
        """Test detection of LaTeX format."""
        assert document_processor.detect_format("thesis.latex") == "latex"
        assert document_processor.detect_format("paper.tex") == "latex"

    def test_detect_unsupported_format(self):
        """Test detection of unsupported format."""
        assert document_processor.detect_format("file.unknown") == ""

    def test_detect_case_insensitive(self):
        """Test format detection is case insensitive."""
        assert document_processor.detect_format("Document.TXT") == "txt"
        assert document_processor.detect_format("README.MD") == "markdown"


class TestProcessorSelection:
    def test_get_text_processor(self):
        """Test getting text processor."""
        processor = document_processor.get_processor("txt")
        assert isinstance(processor, document_processor.TextProcessor)

    def test_get_markdown_processor(self):
        """Test getting markdown processor."""
        processor = document_processor.get_processor("markdown")
        assert isinstance(processor, document_processor.MarkdownProcessor)

    def test_get_html_processor(self):
        """Test getting HTML processor."""
        processor = document_processor.get_processor("html")
        assert isinstance(processor, document_processor.HTMLProcessor)

    def test_get_pdf_processor(self):
        """Test getting PDF processor."""
        processor = document_processor.get_processor("pdf")
        assert isinstance(processor, document_processor.PDFProcessor)

    def test_get_latex_processor(self):
        """Test getting LaTeX processor."""
        processor = document_processor.get_processor("latex")
        assert isinstance(processor, document_processor.LaTeXProcessor)

    def test_get_unknown_processor(self):
        """Test getting processor for unknown format."""
        processor = document_processor.get_processor("unknown")
        assert processor is None


class TestTextProcessor:
    def test_process_simple_text(self):
        """Test processing simple text document."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world. This is a test document.")
            f.flush()
            temp_path = f.name

        try:
            processor = document_processor.TextProcessor()
            chunks = processor.process(temp_path, chunk_size=5, overlap=0)

            assert len(chunks) > 0
            assert all(isinstance(c, document_processor.Chunk) for c in chunks)
            assert all(c.metadata["format"] == "txt" for c in chunks)
            assert all(c.metadata["source"].endswith(".txt") for c in chunks)
        finally:
            Path(temp_path).unlink()

    def test_process_empty_text(self):
        """Test processing empty text document."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            temp_path = f.name

        try:
            processor = document_processor.TextProcessor()
            chunks = processor.process(temp_path, chunk_size=5, overlap=0)
            assert len(chunks) == 0
        finally:
            Path(temp_path).unlink()

    def test_process_with_overlap(self):
        """Test chunking with overlap."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            words = " ".join([f"word{i}" for i in range(20)])
            f.write(words)
            f.flush()
            temp_path = f.name

        try:
            processor = document_processor.TextProcessor()
            chunks = processor.process(temp_path, chunk_size=5, overlap=2)

            assert len(chunks) > 1
            for chunk in chunks:
                assert chunk.metadata["word_count"] <= 5
        finally:
            Path(temp_path).unlink()


class TestMarkdownProcessor:
    def test_process_markdown_sections(self):
        """Test processing markdown with sections."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Header 1\nContent here.\n## Header 2\nMore content.")
            f.flush()
            temp_path = f.name

        try:
            processor = document_processor.MarkdownProcessor()
            chunks = processor.process(temp_path, chunk_size=10, overlap=0)

            assert len(chunks) > 0
            assert all(c.metadata["format"] == "markdown" for c in chunks)
        finally:
            Path(temp_path).unlink()


class TestHTMLProcessor:
    def test_extract_text_from_html(self):
        """Test extracting text from HTML."""
        processor = document_processor.HTMLProcessor()
        html = "<html><body><p>Hello world</p><script>alert('x')</script></body></html>"
        text = processor._extract_text_from_html(html)

        assert "Hello world" in text
        assert "alert" not in text
        assert "script" not in text

    def test_process_html_document(self):
        """Test processing HTML document."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write("<html><body><h1>Title</h1><p>Content here.</p></body></html>")
            f.flush()
            temp_path = f.name

        try:
            processor = document_processor.HTMLProcessor()
            chunks = processor.process(temp_path, chunk_size=5, overlap=0)

            assert len(chunks) > 0
            assert all(c.metadata["format"] == "html" for c in chunks)
        finally:
            Path(temp_path).unlink()


class TestLaTeXProcessor:
    def test_strip_latex_commands(self):
        """Test stripping LaTeX commands."""
        processor = document_processor.LaTeXProcessor()
        latex = r"This is \textbf{content} text with $x^2$ math."
        text = processor._strip_latex_commands(latex)

        assert "content" in text
        assert "textbf" not in text
        assert "$" not in text

    def test_process_latex_document(self):
        """Test processing LaTeX document."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(r"\documentclass{article} \begin{document} Hello world \end{document}")
            f.flush()
            temp_path = f.name

        try:
            processor = document_processor.LaTeXProcessor()
            chunks = processor.process(temp_path, chunk_size=5, overlap=0)

            assert len(chunks) > 0
            assert all(c.metadata["format"] == "latex" for c in chunks)
        finally:
            Path(temp_path).unlink()


class TestChunk:
    def test_chunk_creation(self):
        """Test creating a chunk."""
        chunk = document_processor.Chunk(
            content="Sample content",
            metadata={"source": "test.txt", "page": 1},
        )

        assert chunk.content == "Sample content"
        assert chunk.metadata["source"] == "test.txt"
        assert chunk.metadata["page"] == 1
