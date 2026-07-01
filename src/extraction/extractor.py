"""
File text extraction for PDF, EPUB, DOCX, TXT, and Markdown formats.
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

# Matches "Chapter N: Title", "Chapter N - Title", "Chapter N – Title", etc.
# Tolerates up to 80 leading spaces (centered text in PDFs).
_CHAPTER_RE = re.compile(
    r"^\s{0,80}Chapter\s*:?\s*(\d+)\s*[:\-–—]?\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


class DocReaderError(Exception):
    """User-facing error with a clean message."""
    pass


SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".docx", ".txt", ".md"}


def extract_text(file_path: str) -> str:
    """
    Dispatch to the appropriate extractor based on file extension.
    Returns the body text as a single string.
    Raises DocReaderError for all user-visible failure conditions.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise DocReaderError(
            "Please upload a PDF, EPUB, DOCX, TXT, or MD file."
        )

    try:
        if ext == ".pdf":
            text = _extract_pdf(file_path)
        elif ext == ".epub":
            text = _extract_epub(file_path)
        elif ext == ".docx":
            text = _extract_docx(file_path)
        elif ext == ".txt":
            text = _extract_txt(file_path)
        elif ext == ".md":
            text = _extract_md(file_path)
    except DocReaderError:
        raise
    except Exception as exc:
        raise DocReaderError(
            "Could not read this file. It may be corrupted."
        ) from exc

    if not text or not text.strip():
        raise DocReaderError("No text found in this document.")

    word_count = len(text.split())
    if word_count < 20:
        raise DocReaderError("Document has too little text to convert.")

    return text


# ---------------------------------------------------------------------------
# Format-specific extractors
# ---------------------------------------------------------------------------

def _extract_pdf(path: str) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n\n".join(pages)


def _extract_epub(path: str) -> str:
    """Fallback: merge all spine items into one string (used by extract_text)."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(path, options={"ignore_ncx": True})
    spine_ids = {idref for idref, _ in book.spine}
    parts = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        if item.get_id() not in spine_ids:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        body = soup.find("body")
        if body:
            parts.append(body.get_text(separator=" ", strip=True))

    return "\n\n".join(parts)


def extract_epub_chapters(path: str) -> list:
    """
    Extract EPUB spine items as individual chapters.
    Tries ebooklib first; falls back to direct zipfile parsing for
    non-standard EPUBs that ebooklib cannot open.
    Returns [(chapter_num, title, body_text), ...].
    """
    import zipfile as _zipfile
    try:
        with _zipfile.ZipFile(path) as z:
            z.namelist()
    except _zipfile.BadZipFile as e:
        raise DocReaderError(
            f"This EPUB file is corrupted (bad ZIP structure: {e}). "
            "Please re-download the file from the source and try again."
        )
    except Exception as e:
        raise DocReaderError(
            f"Could not open this EPUB file: {e}. "
            "Try re-downloading it."
        )
    try:
        return _extract_epub_ebooklib(path)
    except DocReaderError:
        raise
    except Exception:
        pass
    # ebooklib failed — try direct ZIP parser
    try:
        return _extract_epub_zipfile(path)
    except DocReaderError:
        raise
    except Exception as exc:
        raise DocReaderError(f"Could not read this EPUB file: {exc}") from exc


def _extract_epub_ebooklib(path: str) -> list:
    """Primary EPUB parser using ebooklib."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(path, options={"ignore_ncx": True})

    spine_ids = {idref for idref, _ in book.spine}
    chapters = []
    chapter_num = 0

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        if item.get_id() not in spine_ids:
            continue

        try:
            soup = BeautifulSoup(item.get_content(), "html.parser")
        except Exception:
            continue

        body = soup.find("body")
        if not body:
            continue

        raw = body.get_text(separator="\n", strip=True)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r"[ \t]+", " ", raw)

        if not raw or len(raw.split()) < 10:
            continue

        sub_chapters = split_chapters(raw)
        if sub_chapters:
            for _, title, body_text in sub_chapters:
                chapter_num += 1
                body_text = _strip_leading_title(body_text, title)
                chapters.append((chapter_num, title, body_text))
            continue

        title = None
        for tag_name in ("h1", "h2", "h3"):
            heading_tag = body.find(tag_name)
            if heading_tag:
                title = re.sub(r"\s+", " ", heading_tag.get_text(strip=True)).strip()
                heading_tag.decompose()
                break

        if title:
            first_p = body.find("p")
            if first_p:
                p_norm = re.sub(r"[^\w\s]", "", first_p.get_text(strip=True)).strip().lower()
                t_norm = re.sub(r"[^\w\s]", "", title).strip().lower()
                if p_norm and t_norm and (p_norm in t_norm or t_norm in p_norm):
                    first_p.decompose()

        text = body.get_text(separator=" ", strip=True)
        text = re.sub(r" {2,}", " ", text)

        if not text or len(text.split()) < 10:
            continue

        chapter_num += 1
        chapters.append((chapter_num, title or f"Section {chapter_num}", text))

    return chapters


def _extract_epub_zipfile(path: str) -> list:
    """Fallback EPUB parser: reads the ZIP directly without ebooklib."""
    import zipfile
    from pathlib import Path as _Path
    from bs4 import BeautifulSoup

    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())

        # Locate OPF via container.xml
        opf_path = None
        if "META-INF/container.xml" in names:
            try:
                container = zf.read("META-INF/container.xml").decode("utf-8", errors="replace")
                m = re.search(r'full-path=["\']([^"\']+\.opf)["\']', container, re.IGNORECASE)
                if m:
                    opf_path = m.group(1)
            except Exception:
                pass
        if not opf_path:
            for n in names:
                if n.endswith(".opf"):
                    opf_path = n
                    break
        if not opf_path:
            raise DocReaderError("Could not find OPF manifest inside EPUB.")

        opf_dir = str(_Path(opf_path).parent)
        opf_content = zf.read(opf_path).decode("utf-8", errors="replace")
        opf_soup = BeautifulSoup(opf_content, "html.parser")

        # Build id → full zip path map
        items = {}
        for item in opf_soup.find_all("item"):
            item_id = item.get("id", "")
            href    = item.get("href", "")
            if item_id and href:
                full = (opf_dir + "/" + href).lstrip("/") if opf_dir != "." else href
                items[item_id] = full

        spine_tag = opf_soup.find("spine")
        if not spine_tag:
            raise DocReaderError("No spine found in EPUB.")
        spine_idrefs = [ir.get("idref") for ir in spine_tag.find_all("itemref") if ir.get("idref")]

        chapters    = []
        chapter_num = 0

        for idref in spine_idrefs:
            zip_entry = items.get(idref)
            if not zip_entry or zip_entry not in names:
                continue
            try:
                content = zf.read(zip_entry)
                soup    = BeautifulSoup(content, "html.parser")
                body    = soup.find("body")
                if not body:
                    continue
                raw = body.get_text(separator="\n", strip=True)
                raw = re.sub(r"\n{3,}", "\n\n", raw)
                raw = re.sub(r"[ \t]+", " ", raw)
                if not raw or len(raw.split()) < 10:
                    continue

                sub_chapters = split_chapters(raw)
                if sub_chapters:
                    for _, title, body_text in sub_chapters:
                        chapter_num += 1
                        body_text = _strip_leading_title(body_text, title)
                        chapters.append((chapter_num, title, body_text))
                    continue

                heading = None
                for tag_name in ("h1", "h2", "h3"):
                    ht = soup.find(tag_name)
                    if ht:
                        heading = ht.get_text(strip=True)
                        break
                chapter_num += 1
                chapters.append((chapter_num, heading or f"Section {chapter_num}", raw))
            except Exception:
                continue

    return chapters


def _extract_docx(path: str) -> str:
    from docx import Document

    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _detect_and_read(path: str) -> str:
    """Try encodings in priority order; return decoded text."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with open(path, encoding=enc) as fh:
                return fh.read()
        except UnicodeDecodeError:
            continue
    # Absolute fallback — replace undecodable bytes
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _extract_txt(path: str) -> str:
    return _detect_and_read(path)


def _extract_md(path: str) -> str:
    return _strip_markdown(_detect_and_read(path))


# ---------------------------------------------------------------------------
# Markdown stripping
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text)
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def _strip_leading_title(body: str, title: str) -> str:
    """Remove a duplicate title line from the very start of body text."""
    if not body or not title:
        return body
    ref = re.sub(r"[^\w\s]", "", title).strip().lower()
    if not ref:
        return body

    nl = body.find("\n")
    first_line = (body[:nl] if nl != -1 else body).strip()
    first_norm = re.sub(r"[^\w\s]", "", first_line).strip().lower()
    if first_norm and (first_norm in ref or ref in first_norm):
        return (body[nl + 1:] if nl != -1 else "").strip()

    return body


def split_chapters(text: str) -> list:
    """
    Detect chapter headings and split text into per-chapter sections.

    Returns a list of (chapter_num, title, body) tuples, or an empty list
    if fewer than 2 chapter headings are found (treat as un-chaptered).
    """
    matches = list(_CHAPTER_RE.finditer(text))
    if len(matches) < 2:
        return []

    result = []
    for i, match in enumerate(matches):
        num = int(match.group(1))
        title = re.sub(r"\s+", " ", match.group(2)).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        if body:
            nl = body.find("\n")
            first_line = (body[:nl] if nl != -1 else body).lstrip()
            if _CHAPTER_RE.match(first_line):
                body = (body[nl + 1:] if nl != -1 else "").strip()

        if body and len(body.split()) >= 10:
            result.append((num, title, body))

    return result
