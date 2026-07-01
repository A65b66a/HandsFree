"""
Book profiler: auto-detect genre, tone, and style from web metadata and AI analysis.

Combines web scraping (NovelUpdates, Goodreads) with Ollama analysis of the
first chapter to build a custom narrator prompt for each book.
Profiles are cached to disk and never re-scraped.
"""

import json
import logging
import re
from pathlib import Path
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

from src.config import PROFILES_DIR, OLLAMA_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)


def scrape_novel_updates(title: str) -> dict:
    """
    Search NovelUpdates for the book title and extract metadata.

    Returns dict with genres, tags, type, and country.
    On ANY error: returns {}.
    """
    try:
        search_url = f"https://www.novelupdates.com/?s={urlencode({'s': title})}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        resp = requests.get(search_url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        result_link = soup.find("a", class_="search")
        if not result_link:
            return {}

        novel_url = result_link.get("href")
        if not novel_url:
            return {}

        resp = requests.get(novel_url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        genres = []
        genre_section = soup.find("div", class_="seriesgenre")
        if genre_section:
            for tag in genre_section.find_all("a"):
                text = tag.get_text(strip=True)
                if text and text != "View all":
                    genres.append(text)

        tags = []
        tags_section = soup.find("div", class_="seriestar")
        if tags_section:
            for tag in tags_section.find_all("a"):
                text = tag.get_text(strip=True)
                if text:
                    tags.append(text)

        novel_type = "Web Novel"
        type_elem = soup.find("div", string=re.compile(r"Type:", re.I))
        if type_elem:
            type_text = type_elem.get_text(strip=True)
            if "Light Novel" in type_text:
                novel_type = "Light Novel"
            elif "Published" in type_text:
                novel_type = "Published"

        country = "unknown"
        country_elem = soup.find("div", string=re.compile(r"Country:", re.I))
        if country_elem:
            country_text = country_elem.get_text(strip=True)
            if "China" in country_text or "Chinese" in country_text:
                country = "China"
            elif "Korea" in country_text or "Korean" in country_text:
                country = "Korea"
            elif "Japan" in country_text or "Japanese" in country_text:
                country = "Japan"
            else:
                country = "Western"

        return {
            "genres": genres,
            "tags": tags,
            "type": novel_type,
            "country": country,
        }

    except Exception:
        return {}


def scrape_goodreads(title: str) -> dict:
    """
    Search Goodreads for the book title and extract metadata.

    Returns dict with genres, avg_rating, and description.
    On ANY error: returns {}.
    """
    try:
        search_url = f"https://www.goodreads.com/search?q={urlencode({'q': title})}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        resp = requests.get(search_url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        book_link = soup.find("a", class_="bookTitle")
        if not book_link:
            return {}

        book_url = book_link.get("href")
        if not book_url:
            return {}

        resp = requests.get(book_url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        genres = []
        shelf_section = soup.find("div", class_="bigBoxContent")
        if shelf_section:
            for link in shelf_section.find_all("a", class_="actionLinkLite"):
                text = link.get_text(strip=True)
                if text and text not in ["more", "less"]:
                    genres.append(text)

        avg_rating = 0.0
        rating_elem = soup.find("div", class_="rating")
        if rating_elem:
            rating_text = rating_elem.get_text(strip=True)
            try:
                avg_rating = float(rating_text.split()[0])
            except (ValueError, IndexError):
                avg_rating = 0.0

        description = ""
        desc_elem = soup.find("span", id=re.compile(r"freeText\d+"))
        if desc_elem:
            description = desc_elem.get_text(strip=True)[:300]

        return {
            "genres": genres,
            "avg_rating": avg_rating,
            "description": description,
        }

    except Exception:
        return {}


def analyze_first_chapter(first_chapter_text: str, title: str) -> dict:
    """
    Send first 2000 chars of chapter to Ollama for tone/style analysis.

    Returns dict with tone, pacing, dialogue_density, action_density,
    emotional_register, protagonist_nature, and narration_style.
    On parse error: returns safe defaults.
    """
    first_2000 = first_chapter_text[:2000]

    prompt = f"""Read this opening from "{title}".
Respond ONLY with a JSON object. No preamble. No explanation.

{{
  "tone": ["word1", "word2", "word3"],
  "protagonist_nature": "one phrase describing protagonist style",
  "pacing": "fast|medium|slow|variable",
  "dialogue_density": "sparse|medium|heavy",
  "action_density": "low|medium|high",
  "emotional_register": "cold|warm|neutral|tense|melancholic",
  "narration_style": "one sentence describing how it feels to read this"
}}

Text:
{first_2000}
"""

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        response_text = resp.json().get("response", "").strip()

        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            return _safe_defaults()

        data = json.loads(json_match.group(0))

        return {
            "tone": data.get("tone", ["neutral"]),
            "protagonist_nature": data.get("protagonist_nature", "unknown"),
            "pacing": data.get("pacing", "medium"),
            "dialogue_density": data.get("dialogue_density", "medium"),
            "action_density": data.get("action_density", "medium"),
            "emotional_register": data.get("emotional_register", "neutral"),
            "narration_style": data.get("narration_style", "Standard prose narration."),
        }

    except Exception:
        return _safe_defaults()


def _safe_defaults() -> dict:
    return {
        "tone": ["neutral"],
        "protagonist_nature": "unknown",
        "pacing": "medium",
        "dialogue_density": "medium",
        "action_density": "medium",
        "emotional_register": "neutral",
        "narration_style": "Standard prose narration.",
    }


def build_book_profile(title: str, first_chapter: str) -> dict:
    """
    Orchestrate all three sources and merge into a single profile.
    Saves the result to PROFILES_DIR/<safe_title>.json.
    """
    logger.info("Scraping NovelUpdates and Goodreads for '%s'...", title)
    web_data = scrape_novel_updates(title)
    gr_data  = scrape_goodreads(title)
    ai_data  = analyze_first_chapter(first_chapter, title)

    profile = {
        "title": title,
        "genres": web_data.get("genres") or gr_data.get("genres") or [],
        "tags": web_data.get("tags", []),
        "country": web_data.get("country", "unknown"),
        "tone": ai_data["tone"],
        "protagonist_nature": ai_data["protagonist_nature"],
        "pacing": ai_data["pacing"],
        "dialogue_density": ai_data["dialogue_density"],
        "action_density": ai_data["action_density"],
        "emotional_register": ai_data["emotional_register"],
        "narration_style": ai_data["narration_style"],
        "description": gr_data.get("description", ""),
    }

    PROFILES_DIR.mkdir(exist_ok=True)
    safe_title = re.sub(r'[\\/:*?"<>|]', "_", title)
    profile_path = PROFILES_DIR / f"{safe_title}.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    return profile


def load_or_build_profile(title: str, first_chapter: str) -> dict:
    """
    Load cached profile from disk if it exists; otherwise build and cache it.
    """
    PROFILES_DIR.mkdir(exist_ok=True)
    safe_title = re.sub(r'[\\/:*?"<>|]', "_", title)
    profile_path = PROFILES_DIR / f"{safe_title}.json"

    if profile_path.exists():
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                logger.debug("Loaded cached profile for '%s'", title)
                return json.load(f)
        except Exception:
            pass

    return build_book_profile(title, first_chapter)


def generate_narrator_prompt(profile: dict) -> str:
    """Generate a complete Ollama system prompt dynamically from a book profile."""
    emotional_register = profile.get("emotional_register", "neutral")
    if emotional_register == "cold":
        voice_character = "The narrator is detached, precise, never emotional."
    elif emotional_register == "warm":
        voice_character = "The narrator is present, empathetic, emotionally engaged."
    elif emotional_register == "tense":
        voice_character = "The narrator is urgent, controlled, always forward-moving."
    elif emotional_register == "melancholic":
        voice_character = "The narrator is reflective, weighted, carries loss."
    else:
        voice_character = "The narrator is clear, measured, lets events speak."

    genres_tags = set(profile.get("genres", []) + profile.get("tags", []))
    genres_tags_lower = {g.lower() for g in genres_tags}

    power_rule = "[beat] before key reveals."
    internal_rule = "Trust sentence structure. Don't over-annotate."

    if any(g in genres_tags_lower for g in ["cultivation", "xianxia", "xuanhuan"]):
        power_rule = "Power reveals: [beat] before — weight of advantage."
        internal_rule = "[cold] for protagonist scheming — voice slows, no gap."
    elif "romance" in genres_tags_lower:
        power_rule = "[slow] for emotional peaks — let feeling land."
        internal_rule = "[breath] generously between emotion and reaction."
    elif any(g in genres_tags_lower for g in ["thriller", "mystery"]):
        power_rule = "[beat] before reveals — tension held one beat longer."
        internal_rule = "Action: no gaps. Tight. Speed is dread."
    elif "horror" in genres_tags_lower:
        power_rule = "[scene] before horror — silence before the drop."
        internal_rule = "[slow] for descriptions of the unknown."
    elif any(g in genres_tags_lower for g in ["literary", "historical"]):
        power_rule = "Rhythm over pause. Let sentence structure breathe."
        internal_rule = "[slow] for philosophical weight. [scene] for era shifts."

    dialogue_density = profile.get("dialogue_density", "medium")
    if dialogue_density == "sparse":
        dialogue_rule = "Dialogue is rare and significant. [breath] after every line."
    elif dialogue_density == "heavy":
        dialogue_rule = "Dialogue flows naturally. Minimal [breath] — only at major shifts."
    else:
        dialogue_rule = "[breath] between dialogue and narration continuation."

    action_density = profile.get("action_density", "medium")
    if action_density == "low":
        action_rule = "Action is rare. When it happens: tight, no gaps."
    elif action_density == "high":
        action_rule = "Combat is relentless. Zero gaps in action. Breathe only at outcomes."
    else:
        action_rule = "Action sequences: no markers. Speed = tension."

    if emotional_register == "cold":
        avoid_list = ["warmth", "enthusiasm", "hesitation markers"]
    elif emotional_register == "warm":
        avoid_list = ["coldness", "detachment", "mechanical pacing"]
    elif emotional_register == "tense":
        avoid_list = ["slowness", "warmth", "reflective pauses"]
    else:
        avoid_list = ["over-annotation", "mechanical timing"]

    prompt = f"""You are narrating "{profile['title']}" as a professional audiobook narrator.

BOOK CHARACTER:
{voice_character}
Narrative style: {profile['narration_style']}
Tone: {', '.join(profile['tone'])}
Protagonist: {profile['protagonist_nature']}

YOUR JOB:
Rewrite each chapter as a complete narrator performance script.
Preserve every event, every name, every piece of dialogue.
Add structural performance markers only where the STORY demands them.

AVAILABLE MARKERS (use sparingly — only where structurally required):
  [scene]  — location/time/perspective shift. 600ms breath. Rare.
  [beat]   — place BEFORE a revelation. Never after. Creates anticipation.
  [breath] — natural pause after dialogue before narration resumes.
  [cold]   — internal thought, cold protagonist voice. No gap — voice quality only.
  [slow]   — weight and gravity. Tragedy, loss, realization.

STRUCTURAL RULES FOR THIS BOOK:
1. {power_rule}
2. {internal_rule}
3. {dialogue_rule}
4. {action_rule}
5. Normal narration: NO markers. Flow continuously. Trust the voice.
6. Enumeration (lists, ranks, facts): rhythmic, tight, no gaps.

NEVER USE MORE THAN 8 MARKERS PER PAGE OF TEXT.
NEVER ADD MARKERS FOR MECHANICAL TIMING — only for story meaning.

AVOID: {', '.join(avoid_list)}

OUTPUT: Only the performance script. No explanations. No preamble.
"""

    return prompt
