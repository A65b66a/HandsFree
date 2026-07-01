"""
Library manager: track converted books and chapter metadata.
Backed by a JSON file in the docreader/library_data/ directory.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from src.config import LIBRARY_DIR

logger = logging.getLogger(__name__)

LIBRARY_DB = LIBRARY_DIR / "library.json"


class LibraryManager:
    def __init__(self):
        self.books = {}
        self._load_library()

    def _load_library(self):
        """Load library data from JSON."""
        LIBRARY_DIR.mkdir(exist_ok=True)
        if LIBRARY_DB.exists():
            try:
                with open(LIBRARY_DB, "r") as f:
                    self.books = json.load(f)
            except Exception as exc:
                logger.warning("Could not load library DB: %s — starting fresh", exc)
                self.books = {}
        else:
            self.books = {}

    def _save_library(self):
        """Save library data to JSON."""
        LIBRARY_DIR.mkdir(exist_ok=True)
        with open(LIBRARY_DB, "w") as f:
            json.dump(self.books, f, indent=2)

    def add_book(self, book_title: str, chapters: list) -> str:
        """
        Add a converted book to the library.

        Args:
            book_title: Name of the book.
            chapters:   List of (chapter_num, chapter_title, wav_path) tuples.

        Returns:
            book_id: unique identifier for the book.
        """
        book_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        chapter_list = []
        for num, title, wav_path in chapters:
            chapter_list.append({
                "num": num,
                "title": title,
                "wav_path": wav_path,
                "duration": self._get_audio_duration(wav_path),
            })

        self.books[book_id] = {
            "title": book_title,
            "created": datetime.now().isoformat(),
            "chapters": chapter_list,
            "total_chapters": len(chapter_list),
        }

        self._save_library()
        return book_id

    def add_single_file(self, book_title: str, wav_path: str) -> str:
        """Add a single-file (non-chapter) book to the library."""
        book_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.books[book_id] = {
            "title": book_title,
            "created": datetime.now().isoformat(),
            "chapters": [{
                "num": 1,
                "title": "Full Book",
                "wav_path": wav_path,
                "duration": self._get_audio_duration(wav_path),
            }],
            "total_chapters": 1,
        }

        self._save_library()
        return book_id

    def get_all_books(self) -> list:
        """Return list of all books with summary metadata."""
        return [
            {
                "book_id": book_id,
                "title": data["title"],
                "chapters": len(data["chapters"]),
                "created": data["created"],
            }
            for book_id, data in self.books.items()
        ]

    def get_book(self, book_id: str) -> dict:
        """Get full book data including all chapters."""
        return self.books.get(book_id)

    def get_chapter_path(self, book_id: str, chapter_num: int) -> str | None:
        """Get the WAV file path for a specific chapter."""
        book = self.books.get(book_id)
        if not book:
            return None
        for chapter in book["chapters"]:
            if chapter["num"] == chapter_num:
                return chapter["wav_path"]
        return None

    def delete_book(self, book_id: str) -> bool:
        """Delete a book record and its chapter WAV files."""
        if book_id not in self.books:
            return False

        for chapter in self.books[book_id]["chapters"]:
            wav_path = chapter["wav_path"]
            try:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass

        del self.books[book_id]
        self._save_library()
        return True

    @staticmethod
    def _get_audio_duration(wav_path: str) -> str:
        """Get audio duration as MM:SS string."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(wav_path)
            duration_ms = len(audio)
            minutes = duration_ms // 60000
            seconds = (duration_ms % 60000) // 1000
            return f"{minutes:02d}:{seconds:02d}"
        except Exception:
            return "00:00"


# Module-level singleton
_library = LibraryManager()


def get_library_manager() -> LibraryManager:
    return _library
