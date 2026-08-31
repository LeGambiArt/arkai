"""Document parsing and chunking for RAG ingestion."""

import html.parser
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    """A processed document chunk with metadata."""

    content: str
    metadata: dict


class DocumentProcessor(ABC):
    """Base class for format-specific document processors."""

    @abstractmethod
    def process(self, file_path: str, chunk_size: int, overlap: int) -> list[Chunk]:
        """Parse and chunk document.

        Args:
            file_path: Path to document file
            chunk_size: Target size for chunks (in tokens/words)
            overlap: Number of tokens/words to overlap between chunks

        Returns:
            List of Chunk objects with content and metadata
        """


class TextProcessor(DocumentProcessor):
    """Processor for plain text files (.txt)."""

    def process(self, file_path: str, chunk_size: int, overlap: int) -> list[Chunk]:
        """Parse and chunk plain text document."""
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        return self._chunk_text(text, file_path, chunk_size, overlap, "txt")

    def _chunk_text(
        self, text: str, file_path: str, chunk_size: int, overlap: int, format: str
    ) -> list[Chunk]:
        """Split text into chunks with overlap."""
        words = text.split()
        chunks = []
        chunk_index = 0
        total_chunks = 0

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            if not chunk_words:
                break
            total_chunks += 1

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            if not chunk_words:
                break

            chunk_content = " ".join(chunk_words)
            metadata = {
                "source": os.path.basename(file_path),
                "format": format,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "word_count": len(chunk_words),
            }

            chunks.append(Chunk(content=chunk_content, metadata=metadata))
            chunk_index += 1

        return chunks


class MarkdownProcessor(DocumentProcessor):
    """Processor for Markdown files (.md)."""

    def process(self, file_path: str, chunk_size: int, overlap: int) -> list[Chunk]:
        """Parse and chunk Markdown document."""
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        chunks = self._extract_sections(text, file_path, chunk_size, overlap)
        return chunks if chunks else self._fallback_chunking(text, file_path, chunk_size, overlap)

    def _extract_sections(
        self, text: str, file_path: str, chunk_size: int, overlap: int
    ) -> list[Chunk]:
        """Extract sections based on markdown headers."""
        lines = text.split("\n")
        sections = []
        current_section = []

        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)

            if header_match:
                if current_section:
                    section_text = "\n".join(current_section).strip()
                    if section_text:
                        sections.append(section_text)
                    current_section = []

                current_section.append(line)
            else:
                current_section.append(line)

        if current_section:
            section_text = "\n".join(current_section).strip()
            if section_text:
                sections.append(section_text)

        chunks = []
        for section_idx, section in enumerate(sections):
            words = section.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i : i + chunk_size]
                if not chunk_words:
                    break

                chunk_content = " ".join(chunk_words)
                metadata = {
                    "source": os.path.basename(file_path),
                    "format": "markdown",
                    "section_index": section_idx,
                    "chunk_index": i // (chunk_size - overlap) if overlap > 0 else i // chunk_size,
                    "word_count": len(chunk_words),
                }

                chunks.append(Chunk(content=chunk_content, metadata=metadata))

        return chunks

    def _fallback_chunking(
        self, text: str, file_path: str, chunk_size: int, overlap: int
    ) -> list[Chunk]:
        """Fallback to word-based chunking if section extraction fails."""
        processor = TextProcessor()
        return processor._chunk_text(text, file_path, chunk_size, overlap, "markdown")


class HTMLProcessor(DocumentProcessor):
    """Processor for HTML files (.html)."""

    def process(self, file_path: str, chunk_size: int, overlap: int) -> list[Chunk]:
        """Parse and chunk HTML document."""
        with open(file_path, encoding="utf-8") as f:
            html_content = f.read()

        text_content = self._extract_text_from_html(html_content)

        processor = TextProcessor()
        chunks = processor._chunk_text(text_content, file_path, chunk_size, overlap, "html")

        for chunk in chunks:
            chunk.metadata["format"] = "html"

        return chunks

    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract plain text from HTML using built-in html.parser."""

        class TextExtractor(html.parser.HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
                self.in_script = False
                self.in_style = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style"):
                    if tag == "script":
                        self.in_script = True
                    elif tag == "style":
                        self.in_style = True

            def handle_endtag(self, tag):
                if tag == "script":
                    self.in_script = False
                elif tag == "style":
                    self.in_style = False

            def handle_data(self, data):
                if not self.in_script and not self.in_style:
                    text = data.strip()
                    if text:
                        self.text.append(text)

        extractor = TextExtractor()
        extractor.feed(html_content)
        text = " ".join(extractor.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class PDFProcessor(DocumentProcessor):
    """Processor for PDF files (.pdf)."""

    def process(self, file_path: str, chunk_size: int, overlap: int) -> list[Chunk]:
        """Parse and chunk PDF document."""
        try:
            import fitz
        except ImportError:
            raise ImportError(
                "pymupdf is required for PDF processing. Install with: pip install pymupdf"
            )

        chunks = []
        chunk_index = 0

        try:
            pdf_document = fitz.open(file_path)
            total_pages = len(pdf_document)

            for page_num in range(total_pages):
                try:
                    page = pdf_document[page_num]
                    text = page.get_text()

                    if not text or not text.strip():
                        continue

                    words = text.split()
                    for i in range(0, len(words), chunk_size - overlap):
                        chunk_words = words[i : i + chunk_size]
                        if not chunk_words:
                            break

                        chunk_content = " ".join(chunk_words)
                        metadata = {
                            "source": os.path.basename(file_path),
                            "format": "pdf",
                            "page": page_num + 1,
                            "total_pages": total_pages,
                            "chunk_index": chunk_index,
                            "word_count": len(chunk_words),
                        }

                        chunks.append(Chunk(content=chunk_content, metadata=metadata))
                        chunk_index += 1
                except Exception:
                    continue

            pdf_document.close()
        except Exception as e:
            raise RuntimeError(f"Error reading PDF file: {e}")

        if not chunks:
            raise RuntimeError("No text content could be extracted from PDF")

        return chunks


class LaTeXProcessor(DocumentProcessor):
    """Processor for LaTeX source files (.latex, .tex)."""

    def process(self, file_path: str, chunk_size: int, overlap: int) -> list[Chunk]:
        """Parse and chunk LaTeX document."""
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        text = self._strip_latex_commands(text)

        processor = TextProcessor()
        chunks = processor._chunk_text(text, file_path, chunk_size, overlap, "latex")

        for chunk in chunks:
            chunk.metadata["format"] = "latex"

        return chunks

    def _strip_latex_commands(self, text: str) -> str:
        """Remove LaTeX commands and markup."""
        text = re.sub(r"\\[a-zA-Z]+\[[^\]]*\]", "", text)
        text = re.sub(r"\$.*?\$", "", text)
        text = re.sub(r"%.*?$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\\[a-zA-Z]+\{(.*?)\}", r"\1", text)
        text = re.sub(r"\\[a-zA-Z]+", "", text)
        text = re.sub(r"\{|\}", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


def detect_format(file_path: str) -> str:
    """Detect document format from file extension."""
    _, ext = os.path.splitext(file_path.lower())
    ext = ext.lstrip(".")

    format_map = {
        "txt": "txt",
        "md": "markdown",
        "html": "html",
        "htm": "html",
        "pdf": "pdf",
        "latex": "latex",
        "tex": "latex",
    }

    return format_map.get(ext, "")


def get_processor(format: str) -> Optional[DocumentProcessor]:
    """Get processor instance for detected format."""
    processors = {
        "txt": TextProcessor(),
        "markdown": MarkdownProcessor(),
        "html": HTMLProcessor(),
        "pdf": PDFProcessor(),
        "latex": LaTeXProcessor(),
    }

    return processors.get(format)
