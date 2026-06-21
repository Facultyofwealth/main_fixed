# ============================================================
#  In The Beginning — FastAPI Backend  (Deepgram-first edition)
#  Run:  uvicorn main_fixed:app --reload --port 8000
# ============================================================

import os, sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── Load .env BEFORE anything else ───────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_path   = os.path.join(_script_dir, ".env")
if not os.path.exists(_env_path):
    _env_path = os.path.join(os.getcwd(), ".env")

from dotenv import load_dotenv
load_dotenv(dotenv_path=_env_path, override=True)
print(f"📄 .env loaded from: {_env_path}  (exists={os.path.exists(_env_path)})")

import time, json, asyncio, re, tempfile, io, zipfile, socket
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Tuple
from urllib.parse import quote

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.websockets import WebSocketState

# ── Deepgram key ──────────────────────────────────────────────
SERVER_DG_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()
if SERVER_DG_KEY and SERVER_DG_KEY != "your_deepgram_api_key_here":
    print(f"🔑 Server DEEPGRAM_API_KEY found: {SERVER_DG_KEY[:8]}…")
else:
    SERVER_DG_KEY = ""
    print("ℹ  No server Deepgram key — clients may supply their own.")

# ── Local vector search (FAISS + BGE embeddings) ──────────────
#  Lane 1: direct Bible references are caught by regex/state and never sent
#          into the semantic model.
#  Lane 2: normal paraphrases are embedded locally with bge-small-en-v1.5 and
#          searched in FAISS against verse text only.
# ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("ITB_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip()
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
VECTOR_MIN_SCORE = float(os.getenv("ITB_VECTOR_MIN_SCORE", "0.42"))
VECTOR_READY = False
VECTOR_STATUS = "not initialized"

try:
    import faiss
except ImportError:
    faiss = None
    VECTOR_STATUS = "faiss-cpu is not installed"

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    if faiss is not None:
        VECTOR_STATUS = "sentence-transformers is not installed"

# Kept false so any old compatibility helper cannot call cloud ranking.
OPENAI_AVAILABLE = False
_openai_client = None

# ── Whisper (offline fallback) ────────────────────────────────
# Model is NOT loaded at startup — it loads lazily the first time a Whisper
# session begins (see _ensure_whisper_model), keeping boot memory light.
WHISPER_AVAILABLE = False
whisper_model     = None

try:
    import whisper as _whisper_lib
    WHISPER_AVAILABLE = True          # library is present; model loads on demand
    print("✅ Whisper library found — model will load on first use")
except ImportError:
    _whisper_lib = None
    print("⚠  Whisper not installed — run: pip install openai-whisper")

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title="In The Beginning API", version="6.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _find_frontend() -> Optional[str]:
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "In_the_Beginning.html"),
        os.path.join(base, "In the Beginning.html"),
        os.path.join(os.getcwd(), "In_the_Beginning.html"),
        os.path.join(os.getcwd(), "In the Beginning.html"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"🌐 Frontend found: {p}")
            return p
    print("⚠  Frontend HTML not found")
    return None

FRONTEND_FILE = _find_frontend()
if FRONTEND_FILE:
    @app.get("/")
    def serve_frontend():
        return FileResponse(FRONTEND_FILE)

# ── Constants ─────────────────────────────────────────────────
TOP_K     = 5   # retrieve top-5 from FAISS/keyword; display top-3
MIN_WORDS = 3
APP_PORT = int(os.getenv("ITB_APP_PORT", "8000"))
LAN_PROXY_PORT_START = int(os.getenv("ITB_REMOTE_PORT", "8001"))
LAN_PROXY_PORT_ACTIVE: Optional[int] = None
LAN_PROXY_SERVER: Optional[asyncio.AbstractServer] = None

# Deepgram LiveOptions tuned for fast visible words and quick final chunks.
# Endpointing: ms of silence Deepgram waits before emitting speech_final.
# 250 ms catches natural pauses without cutting mid-sentence.
DEEPGRAM_ENDPOINT_MS      = 250
# UtteranceEnd: Deepgram's VAD window. 2400 ms gives a fast preacher
# time to breathe mid-verse without splitting the utterance.
DEEPGRAM_UTTERANCE_END_MS = 2400

NEXT_VERSE_RE = re.compile(
    r"\b(next\s*verse|next\s*scripture|move\s*on|go\s*to\s*next|advance\s+verse|proceed|the\s*next\s*one)\b",
    re.IGNORECASE,
)
PREVIOUS_VERSE_RE = re.compile(
    r"\b(previous\s*verse|prev\s*verse|prior\s*verse|last\s*verse|go\s*back|back\s*one|the\s*previous\s*one)\b",
    re.IGNORECASE,
)

TRANSLATION_ALIASES = {
    "kjv": "KJV",
    "king james": "KJV",
    "king james version": "KJV",
    "authorized version": "KJV",
    "niv": "NIV",
    "new international": "NIV",
    "new international version": "NIV",
    "esv": "ESV",
    "english standard": "ESV",
    "english standard version": "ESV",
    "nkjv": "NKJV",
    "new king james": "NKJV",
    "new king james version": "NKJV",
    "nlt": "NLT",
    "new living": "NLT",
    "new living translation": "NLT",
    "amp": "AMP",
    "amplified": "AMP",
    "amplified bible": "AMP",
    "msg": "MSG",
    "message": "MSG",
    "the message": "MSG",
    "csb": "CSB",
    "christian standard": "CSB",
    "christian standard bible": "CSB",
    "nasb": "NASB",
    "new american standard": "NASB",
    "rsv": "RSV",
    "web": "WEB",
    "world english bible": "WEB",
    "ylt": "YLT",
    "young literal": "YLT",
}

# ── Scripture feed ────────────────────────────────────────────
scripture_feed: deque = deque(maxlen=200)

def add_to_feed(verse: Dict, match_type: str = "match"):
    entry = {**verse, "type": match_type, "timestamp": time.time()}
    if scripture_feed and scripture_feed[-1].get("ref") == verse.get("ref"):
        return
    scripture_feed.append(entry)

# ── Bible Index ───────────────────────────────────────────────
STOP_WORDS = {
    "the","a","an","and","or","of","in","to","is","it","he","she","they",
    "i","that","this","was","be","are","his","her","not","but","for","with"
}
BOOK_ALIASES = {
    "psalm": "Psalms",
    "psalms": "Psalms",
    "song of songs": "Song of Solomon",
    "songs of solomon": "Song of Solomon",
    "canticles": "Song of Solomon",
    "revelations": "Revelation",
}

BOOK_SHORT_CODES = {
    # ── Old Testament ────────────────────────────────────────────
    "gen": "Genesis",       "genesis": "Genesis",
    "exo": "Exodus",        "exodus": "Exodus",
    "lev": "Leviticus",     "leviticus": "Leviticus",
    "num": "Numbers",       "numbers": "Numbers",
    "deu": "Deuteronomy",   "deuteronomy": "Deuteronomy",   "deut": "Deuteronomy",
    "jos": "Joshua",        "joshua": "Joshua",
    "jdg": "Judges",        "judg": "Judges",               "judges": "Judges",
    "rut": "Ruth",          "ruth": "Ruth",
    "1sa": "1 Samuel",      "1 sam": "1 Samuel",            "1samuel": "1 Samuel",
    "2sa": "2 Samuel",      "2 sam": "2 Samuel",            "2samuel": "2 Samuel",
    "1ki": "1 Kings",       "1 kin": "1 Kings",             "1kings": "1 Kings",
    "2ki": "2 Kings",       "2 kin": "2 Kings",             "2kings": "2 Kings",
    "1ch": "1 Chronicles",  "1 chr": "1 Chronicles",        "1chronicles": "1 Chronicles",
    "2ch": "2 Chronicles",  "2 chr": "2 Chronicles",        "2chronicles": "2 Chronicles",
    "ezr": "Ezra",          "ezra": "Ezra",
    "neh": "Nehemiah",      "nehemiah": "Nehemiah",
    "est": "Esther",        "esther": "Esther",
    "job": "Job",
    "psa": "Psalms",        "pss": "Psalms",                "psalms": "Psalms",   "psalm": "Psalms",   "ps": "Psalms",
    "pro": "Proverbs",      "proverbs": "Proverbs",         "prov": "Proverbs",
    "ecc": "Ecclesiastes",  "ecclesiastes": "Ecclesiastes", "eccl": "Ecclesiastes",
    "son": "Song of Solomon","sng": "Song of Solomon",       "sos": "Song of Solomon",
    "isa": "Isaiah",        "isaiah": "Isaiah",
    "jer": "Jeremiah",      "jeremiah": "Jeremiah",
    "lam": "Lamentations",  "lamentations": "Lamentations",
    "eze": "Ezekiel",       "ezk": "Ezekiel",               "ezekiel": "Ezekiel",
    "dan": "Daniel",        "daniel": "Daniel",
    "hos": "Hosea",         "hosea": "Hosea",
    "joe": "Joel",          "joel": "Joel",
    "amo": "Amos",          "amos": "Amos",
    "oba": "Obadiah",       "obadiah": "Obadiah",
    "jon": "Jonah",         "jonah": "Jonah",
    "mic": "Micah",         "micah": "Micah",
    "nah": "Nahum",         "nahum": "Nahum",
    "hab": "Habakkuk",      "habakkuk": "Habakkuk",
    "zep": "Zephaniah",     "zephaniah": "Zephaniah",
    "hag": "Haggai",        "haggai": "Haggai",
    "zec": "Zechariah",     "zechariah": "Zechariah",
    "mal": "Malachi",       "malachi": "Malachi",
    # ── New Testament ────────────────────────────────────────────
    "mat": "Matthew",       "matthew": "Matthew",           "matt": "Matthew",
    "mar": "Mark",          "mark": "Mark",                 "mrk": "Mark",
    "luk": "Luke",          "luke": "Luke",
    "joh": "John",          "john": "John",
    "act": "Acts",          "acts": "Acts",
    "rom": "Romans",        "romans": "Romans",
    "1co": "1 Corinthians", "1 cor": "1 Corinthians",       "1corinthians": "1 Corinthians",
    "2co": "2 Corinthians", "2 cor": "2 Corinthians",       "2corinthians": "2 Corinthians",
    "gal": "Galatians",     "galatians": "Galatians",
    "eph": "Ephesians",     "ephesians": "Ephesians",
    "phi": "Philippians",   "php": "Philippians",           "philippians": "Philippians",  "phil": "Philippians",
    "col": "Colossians",    "colossians": "Colossians",
    "1th": "1 Thessalonians","1 the": "1 Thessalonians",    "1thessalonians": "1 Thessalonians",
    "2th": "2 Thessalonians","2 the": "2 Thessalonians",    "2thessalonians": "2 Thessalonians",
    "1ti": "1 Timothy",     "1 tim": "1 Timothy",           "1timothy": "1 Timothy",
    "2ti": "2 Timothy",     "2 tim": "2 Timothy",           "2timothy": "2 Timothy",
    "tit": "Titus",         "titus": "Titus",
    "phm": "Philemon",      "philemon": "Philemon",
    "heb": "Hebrews",       "hebrews": "Hebrews",
    "jam": "James",         "jas": "James",                 "james": "James",
    "1pe": "1 Peter",       "1 pet": "1 Peter",             "1peter": "1 Peter",
    "2pe": "2 Peter",       "2 pet": "2 Peter",             "2peter": "2 Peter",
    "1jo": "1 John",        "1 joh": "1 John",              "1john": "1 John",
    "2jo": "2 John",        "2 joh": "2 John",              "2john": "2 John",
    "3jo": "3 John",        "3 joh": "3 John",              "3john": "3 John",
    "jud": "Jude",          "jude": "Jude",
    "rev": "Revelation",    "revelation": "Revelation",     "revelations": "Revelation",
}
BOOK_ALIASES.update(BOOK_SHORT_CODES)

def _tokenize_words(text: str) -> List[str]:
    return [word for word in re.sub(r"[^\w\s]", " ", text.lower()).split() if word]

def canonicalize_book_name(book: str) -> str:
    clean = re.sub(r"\s+", " ", (book or "").strip())
    if not clean:
        return clean
    return BOOK_ALIASES.get(clean.lower(), clean)

def normalise_reference_text(ref: str) -> str:
    clean = re.sub(r"\s+", " ", (ref or "").strip())
    match = re.match(r"^(.+?)\s+(\d+)(?::(\d+))?$", clean, re.IGNORECASE)
    if not match:
        return clean
    book, chapter, verse = match.groups()
    normalized = f"{canonicalize_book_name(book)} {int(chapter)}"
    if verse is not None:
        normalized += f":{int(verse)}"
    return normalized

class BibleIndex:
    def __init__(self):
        self.verses: List[Dict] = []
        self._last_idx: Optional[int] = None
        self._ref_lookup: Dict[str, Tuple[int, Dict]] = {}
        self._ref_lookup_nospace: Dict[str, Tuple[int, Dict]] = {}
        self._idx_by_ref: Dict[str, int] = {}
        self._book_aliases: Dict[str, str] = {}
        self._canonical_texts: List[str] = []
        self._search_rows = []
        self._keyword_cache: Dict[Tuple[str, int], List[Dict]] = {}
        self._embedding_model = None
        self._faiss_index = None
        self._vector_ready = False
        self._vector_status = "not built"

    def load(self):
        candidates = [
            "kjv.json/kjv-master/json/verses-1769.json",
            "kjv.json", "verses-1769.json", "bible.json",
        ]
        for path in candidates:
            verses = self._try_load(path)
            if verses:
                self.verses = verses
                self._rebuild_lookup_maps()
                print(f"✅ Loaded {len(self.verses)} verses from {path}")
                return
        self.verses = self._sample()
        self._rebuild_lookup_maps()
        print(f"ℹ  Using {len(self.verses)} built-in sample verses")

    def _try_load(self, filepath: str) -> List[Dict]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                bible = json.load(f)
            if isinstance(bible, dict) and all(":" in k for k in list(bible)[:5]):
                return [{"ref": r, "text": t.strip()} for r, t in bible.items() if t.strip()]
            verses = []
            for book, data in (bible.items() if isinstance(bible, dict) else []):
                if not isinstance(data, list): continue
                for ch_idx, ch in enumerate(data):
                    if isinstance(ch, dict):
                        for v in ch.get("verses", []):
                            ref  = f"{book} {ch_idx+1}:{v.get('verse','?')}"
                            text = v.get("text", "").strip()
                            if text: verses.append({"ref": ref, "text": text})
            return verses
        except Exception:
            return []

    def _sample(self) -> List[Dict]:
        return [
            {"ref": "Genesis 1:1",        "text": "In the beginning God created the heaven and the earth."},
            {"ref": "Genesis 1:2",        "text": "And the earth was without form, and void; and darkness was upon the face of the deep."},
            {"ref": "Psalm 23:1",         "text": "The LORD is my shepherd; I shall not want."},
            {"ref": "Psalm 23:2",         "text": "He maketh me to lie down in green pastures: he leadeth me beside the still waters."},
            {"ref": "Psalm 23:3",         "text": "He restoreth my soul: he leadeth me in the paths of righteousness for his name's sake."},
            {"ref": "John 3:16",          "text": "For God so loved the world, that he gave his only begotten Son, that whosoever believeth in him should not perish, but have everlasting life."},
            {"ref": "John 3:17",          "text": "For God sent not his Son into the world to condemn the world; but that the world through him might be saved."},
            {"ref": "Proverbs 3:5",       "text": "Trust in the LORD with all thine heart; and lean not unto thine own understanding."},
            {"ref": "Proverbs 3:6",       "text": "In all thy ways acknowledge him, and he shall direct thy paths."},
            {"ref": "Romans 8:28",        "text": "And we know that all things work together for good to them that love God, to them who are the called according to his purpose."},
            {"ref": "Philippians 4:13",   "text": "I can do all things through Christ which strengtheneth me."},
            {"ref": "Isaiah 40:31",       "text": "But they that wait upon the LORD shall renew their strength; they shall mount up with wings as eagles; they shall run, and not be weary."},
            {"ref": "Jeremiah 29:11",     "text": "For I know the thoughts that I think toward you, saith the LORD, thoughts of peace, and not of evil, to give you an expected end."},
            {"ref": "Matthew 28:19",      "text": "Go ye therefore, and teach all nations, baptizing them in the name of the Father, and of the Son, and of the Holy Ghost."},
            {"ref": "John 1:1",           "text": "In the beginning was the Word, and the Word was with God, and the Word was God."},
            {"ref": "Romans 10:9",        "text": "That if thou shalt confess with thy mouth the Lord Jesus, and shalt believe in thine heart that God hath raised him from the dead, thou shalt be saved."},
            {"ref": "Ephesians 2:8",      "text": "For by grace are ye saved through faith; and that not of yourselves: it is the gift of God."},
            {"ref": "1 Corinthians 13:4", "text": "Charity suffereth long, and is kind; charity envieth not; charity vaunteth not itself, is not puffed up."},
            {"ref": "Hebrews 11:1",       "text": "Now faith is the substance of things hoped for, the evidence of things not seen."},
            {"ref": "Matthew 6:33",       "text": "But seek ye first the kingdom of God, and his righteousness; and all these things shall be added unto you."},
            {"ref": "Psalm 46:1",         "text": "God is our refuge and strength, a very present help in trouble."},
            {"ref": "Isaiah 53:5",        "text": "But he was wounded for our transgressions, he was bruised for our iniquities: the chastisement of our peace was upon him; and with his stripes we are healed."},
            {"ref": "2 Timothy 3:16",     "text": "All scripture is given by inspiration of God, and is profitable for doctrine, for reproof, for correction, for instruction in righteousness."},
            {"ref": "Psalms 119:105",     "text": "Thy word is a lamp unto my feet, and a light unto my path."},
            {"ref": "John 14:6",          "text": "Jesus saith unto him, I am the way, the truth, and the life: no man cometh unto the Father, but by me."},
            {"ref": "Mark 11:23",         "text": "For verily I say unto you, That whosoever shall say unto this mountain, Be thou removed, and be thou cast into the sea; and shall not doubt in his heart, but shall believe that those things which he saith shall come to pass; he shall have whatsoever he saith."},
            {"ref": "1 Thessalonians 4:16","text": "For the Lord himself shall descend from heaven with a shout, with the voice of the archangel, and with the trump of God: and the dead in Christ shall rise first."},
            {"ref": "2 Corinthians 5:7",  "text": "For we walk by faith, not by sight."},
            {"ref": "Galatians 2:20",     "text": "I am crucified with Christ: nevertheless I live; yet not I, but Christ liveth in me."},
        ]

    def keyword_search(self, query: str, k: int = TOP_K) -> List[Dict]:
        q_words = [word for word in _tokenize_words(query) if word not in STOP_WORDS]
        if not q_words:
            return []
        cache_key = (re.sub(r"\s+", " ", query.lower()).strip(), int(k))
        cached = self._keyword_cache.get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]

        phrase = " ".join(q_words)
        scored = []
        for i, v in enumerate(self.verses):
            if i < len(self._search_rows):
                searchable, word_set, v_words = self._search_rows[i]
            else:
                searchable = f"{v['ref']} {v['text']}".lower()
                v_words = tuple(dict.fromkeys(_tokenize_words(searchable)))
                word_set = set(v_words)
            exact_hits = 0
            prefix_hits = 0
            substring_hits = 0
            for word in q_words:
                if word in word_set:
                    exact_hits += 1
                elif any(candidate.startswith(word) for candidate in v_words):
                    prefix_hits += 1
                elif word in searchable:
                    substring_hits += 1
            total_hits = exact_hits + prefix_hits + substring_hits
            if total_hits:
                score = (
                    exact_hits +
                    (prefix_hits * 0.75) +
                    (substring_hits * 0.35)
                ) / max(len(q_words), 1)
                if phrase and phrase in searchable:
                    score += 0.35
                scored.append({**v, "score": round(score, 3), "_idx": i})
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:k]
        if len(self._keyword_cache) >= 256:
            self._keyword_cache.pop(next(iter(self._keyword_cache)), None)
        self._keyword_cache[cache_key] = [dict(item) for item in results]
        return results

    def build_vector_index(self):
        global VECTOR_READY, VECTOR_STATUS
        self._vector_ready = False
        self._faiss_index = None
        if faiss is None or SentenceTransformer is None:
            self._vector_status = VECTOR_STATUS
            VECTOR_READY = False
            return
        if not self.verses:
            self._vector_status = "Bible index is empty"
            VECTOR_STATUS = self._vector_status
            VECTOR_READY = False
            return

        # ── Cache paths (live next to this script) ────────────────
        _base = os.path.dirname(os.path.abspath(__file__))
        _faiss_path = os.path.join(_base, "bible_index.faiss")
        _emb_path   = os.path.join(_base, "bible_embeddings.npy")
        # Cache is keyed by (model name, verse count) so a Bible update
        # or model change automatically invalidates and rebuilds.
        _cache_meta_path = os.path.join(_base, "bible_cache_meta.json")

        def _cache_valid() -> bool:
            if not (os.path.exists(_faiss_path) and os.path.exists(_emb_path)
                    and os.path.exists(_cache_meta_path)):
                return False
            try:
                with open(_cache_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return (meta.get("model") == EMBEDDING_MODEL_NAME
                        and meta.get("verse_count") == len(self.verses))
            except Exception:
                return False

        try:
            import numpy as np

            if _cache_valid():
                # ── Fast path: load everything from disk ─────────
                print(f"⚡ Loading cached FAISS index & embeddings…")
                # Silence HuggingFace progress output
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                os.environ["TRANSFORMERS_VERBOSITY"] = "error"
                os.environ["SENTENCE_TRANSFORMERS_VERBOSITY"] = "0"
                self._embedding_model = SentenceTransformer(
                    EMBEDDING_MODEL_NAME, local_files_only=False
                )
                self._faiss_index = faiss.read_index(_faiss_path)
                self._vector_ready = True
                self._vector_status = f"ready: {EMBEDDING_MODEL_NAME} ({self._faiss_index.ntotal} verses) [cached]"
                VECTOR_READY = True
                VECTOR_STATUS = self._vector_status
                print(f"✅ FAISS vector index ready (from cache): {self._faiss_index.ntotal} verses")
                return

            # ── Slow path: build from scratch and save cache ──────
            print(f"🔨 Building FAISS index for {len(self.verses)} verses (first run — will be cached)…")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            os.environ["SENTENCE_TRANSFORMERS_VERBOSITY"] = "0"
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            corpus = [v.get("text", "") for v in self.verses]
            embeddings = self._embedding_model.encode(
                corpus,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embeddings = embeddings.astype("float32", copy=False)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self._faiss_index = index
            self._vector_ready = True
            self._vector_status = f"ready: {EMBEDDING_MODEL_NAME} ({index.ntotal} verses)"
            VECTOR_READY = True
            VECTOR_STATUS = self._vector_status
            print(f"✅ FAISS vector index ready: {index.ntotal} verses")

            # ── Persist to disk ───────────────────────────────────
            try:
                faiss.write_index(index, _faiss_path)
                np.save(_emb_path, embeddings)
                with open(_cache_meta_path, "w", encoding="utf-8") as f:
                    json.dump({"model": EMBEDDING_MODEL_NAME, "verse_count": len(self.verses)}, f)
                print(f"💾 FAISS cache saved → {_faiss_path}")
            except Exception as save_err:
                print(f"⚠  Could not save FAISS cache: {save_err}")

        except Exception as e:
            self._embedding_model = None
            self._faiss_index = None
            self._vector_ready = False
            self._vector_status = f"vector index unavailable: {e}"
            VECTOR_READY = False
            VECTOR_STATUS = self._vector_status
            print(f"FAISS vector index unavailable: {e}")

    def vector_status(self) -> Dict:
        return {
            "ready": self._vector_ready,
            "model": EMBEDDING_MODEL_NAME,
            "status": self._vector_status,
        }

    def vector_search(self, query: str, k: int = TOP_K) -> List[Dict]:
        if not self._vector_ready or not self._embedding_model or self._faiss_index is None:
            return self.keyword_search(query, k)
        clean_query = re.sub(r"\s+", " ", (query or "").strip())
        if not clean_query:
            return []
        try:
            query_vec = self._embedding_model.encode(
                [BGE_QUERY_PREFIX + clean_query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32", copy=False)
            scores, ids = self._faiss_index.search(query_vec, max(k, 1))
            results: List[Dict] = []
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0 or idx >= len(self.verses):
                    continue
                verse = self.verses[int(idx)]
                results.append({**verse, "score": round(float(score), 3), "_idx": int(idx), "search_lane": "faiss"})
            return results
        except Exception as e:
            print(f"FAISS search error; falling back to keyword search: {e}")
            return self.keyword_search(query, k)

    def _lookup_key(self, ref: str) -> str:
        return re.sub(r"\s+", " ", normalise_reference_text(ref).lower()).strip()

    def _rebuild_lookup_maps(self):
        self._ref_lookup.clear()
        self._ref_lookup_nospace.clear()
        self._idx_by_ref.clear()
        self._book_aliases.clear()
        self._canonical_texts = []
        self._search_rows = []
        self._keyword_cache.clear()
        for i, verse in enumerate(self.verses):
            self._canonical_texts.append(_canonical_text(verse.get("text", "")))
            raw_ref = verse.get("ref", "")
            searchable = f"{raw_ref} {verse.get('text', '')}".lower()
            word_list = tuple(dict.fromkeys(_tokenize_words(searchable)))
            self._search_rows.append((searchable, set(word_list), word_list))
            for key in {
                self._lookup_key(raw_ref),
                re.sub(r"\s+", " ", raw_ref.lower()).strip(),
            }:
                if not key:
                    continue
                self._ref_lookup[key] = (i, verse)
                self._ref_lookup_nospace[key.replace(" ", "")] = (i, verse)
            canonical_key = self._lookup_key(raw_ref)
            if canonical_key:
                self._idx_by_ref[canonical_key] = i
            parsed = self.split_ref(raw_ref)
            if parsed:
                self._add_book_aliases(parsed[0])

    def _add_book_aliases(self, book: str):
        aliases = {book, book.lower(), canonicalize_book_name(book)}
        for alias, canonical in BOOK_ALIASES.items():
            if canonical.lower() == book.lower():
                aliases.add(alias)
        words_only = re.sub(r"[^A-Za-z\s]", "", book).split()
        if words_only:
            short_code = words_only[-1][:3].lower()
            if BOOK_ALIASES.get(short_code, book).lower() == book.lower():
                aliases.add(short_code)
        number_match = re.match(r"^([1-3])\s+(.+)$", book)
        if number_match:
            number, rest = number_match.groups()
            words = {"1": ("first", "one", "1st"), "2": ("second", "two", "2nd"), "3": ("third", "three", "3rd")}
            aliases.update({f"{word} {rest}" for word in words[number]})
            aliases.add(f"{number}{rest}")
            aliases.add(f"{number} {rest[:3]}")
            aliases.add(f"{number}{rest[:3]}")
        for alias in aliases:
            clean = re.sub(r"\s+", " ", alias.lower()).strip()
            if clean:
                self._book_aliases[clean] = book

    def book_aliases(self) -> List[Tuple[str, str]]:
        return sorted(self._book_aliases.items(), key=lambda item: len(item[0]), reverse=True)

    def _with_idx(self, idx: int, verse: Dict, score: float = 1.0) -> Dict:
        return {**verse, "score": score, "_idx": idx}

    def find_by_ref(self, ref: str) -> Optional[Dict]:
        for candidate in (ref, normalise_reference_text(ref)):
            key = self._lookup_key(candidate)
            found = self._ref_lookup.get(key)
            if found:
                idx, verse = found
                return self._with_idx(idx, verse)
            found = self._ref_lookup_nospace.get(key.replace(" ", ""))
            if found:
                idx, verse = found
                return self._with_idx(idx, verse)
        return None

    def find_by_parts(self, book: str, chapter: int, verse: int) -> Optional[Dict]:
        return self.find_by_ref(f"{canonicalize_book_name(book)} {int(chapter)}:{int(verse)}")

    def split_ref(self, ref: str) -> Optional[Tuple[str, int, int]]:
        match = re.match(r"^(.+?)\s+(\d+):(\d+)$", (ref or "").strip(), re.IGNORECASE)
        if not match:
            return None
        book, chapter, verse = match.groups()
        return canonicalize_book_name(book), int(chapter), int(verse)

    def get_adjacent_to_ref(self, ref: str, offset: int) -> Optional[Dict]:
        current = self.find_by_ref(ref)
        if not current:
            return None
        idx = current.get("_idx")
        if idx is None:
            return None
        next_idx = idx + offset
        if 0 <= next_idx < len(self.verses):
            self._last_idx = next_idx
            return self._with_idx(next_idx, self.verses[next_idx])
        return None

    def find_verbatim_phrase(self, query: str) -> Optional[Dict]:
        canon_query = _canonical_text(query)
        words = canon_query.split()
        if len(words) < 5:
            return None
        for idx, canon_text in enumerate(self._canonical_texts):
            if canon_query and canon_query in canon_text:
                return self._with_idx(idx, self.verses[idx], score=0.98)
        return None

    def get_next_verse(self) -> Optional[Dict]:
        if self._last_idx is None: return None
        nxt = self._last_idx + 1
        if nxt < len(self.verses):
            self._last_idx = nxt
            v = self.verses[nxt]
            return {**v, "score": 1.0, "_idx": nxt}
        return None

    def set_manual_index(self, ref: str):
        verse = self.find_by_ref(ref)
        if verse:
            self._last_idx = verse.get("_idx")
            print(f"Index set to [{self._last_idx}] {verse['ref']}")
            return

bible = BibleIndex()

# ── Shared session state ───────────────────────────────────────
# One module-level SessionState is kept in sync by every code path that
# displays a verse (/ws/live transcription + /ws/control remote commands).
# This ensures remote NEXT/PREV always move relative to the *last displayed*
# verse, even when the preacher has jumped to a completely new reference.
global_session: "SessionState" = None   # populated after SessionState is defined

async def _copy_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        while True:
            chunk = await reader.read(64 * 1024)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

async def _handle_lan_proxy(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
):
    try:
        server_reader, server_writer = await asyncio.open_connection("127.0.0.1", APP_PORT)
    except Exception:
        try:
            client_writer.close()
            await client_writer.wait_closed()
        except Exception:
            pass
        return

    await asyncio.gather(
        _copy_stream(client_reader, server_writer),
        _copy_stream(server_reader, client_writer),
        return_exceptions=True,
    )

async def _ensure_lan_proxy():
    global LAN_PROXY_PORT_ACTIVE, LAN_PROXY_SERVER
    if LAN_PROXY_SERVER:
        return

    for port in range(LAN_PROXY_PORT_START, LAN_PROXY_PORT_START + 10):
        try:
            LAN_PROXY_SERVER = await asyncio.start_server(_handle_lan_proxy, "0.0.0.0", port)
            LAN_PROXY_PORT_ACTIVE = port
            print(f"Remote LAN proxy ready: 0.0.0.0:{port} -> 127.0.0.1:{APP_PORT}")
            asyncio.create_task(LAN_PROXY_SERVER.serve_forever())
            return
        except OSError:
            continue

    print("Remote LAN proxy could not start; run uvicorn with --host 0.0.0.0 for phone remote access.")

@app.on_event("startup")
async def startup():
    bible.load()
    # FAISS index is NOT loaded at startup to stay within 512MB free-tier RAM.
    # It loads lazily the first time transcription begins (see _ensure_vector_index).
    await _ensure_lan_proxy()
    dg  = f"✅ ({SERVER_DG_KEY[:8]}…)" if SERVER_DG_KEY else "⚠  no key"
    wh  = "✅ lib ready" if WHISPER_AVAILABLE else "❌ not installed"
    print(f"🚀 Ready | Deepgram: {dg} | Whisper: {wh} | Both Whisper & FAISS load lazily on first use")
    print(f"📖 Frontend: {FRONTEND_FILE or 'NOT FOUND'}")

_vector_index_loaded = False
_vector_index_lock = None   # created lazily inside the event loop (asyncio.Lock() at module level crashes Python 3.12+)

async def _ensure_vector_index():
    """Load FAISS + embedding model once, on first transcription start."""
    global _vector_index_loaded, _vector_index_lock
    if _vector_index_loaded:
        return
    if _vector_index_lock is None:
        _vector_index_lock = asyncio.Lock()
    async with _vector_index_lock:
        if _vector_index_loaded:
            return
        print("⏳ Loading FAISS vector index on first use…")
        await asyncio.to_thread(bible.build_vector_index)
        _vector_index_loaded = True
        print("✅ FAISS vector index ready")

_whisper_model_loaded = False
_whisper_model_lock = None  # created lazily inside the event loop

async def _ensure_whisper_model():
    """Load Whisper 'tiny' model once, on first Whisper session start."""
    global whisper_model, _whisper_model_loaded, _whisper_model_lock
    if _whisper_model_loaded:
        return
    if _whisper_model_lock is None:
        _whisper_model_lock = asyncio.Lock()
    async with _whisper_model_lock:
        if _whisper_model_loaded:
            return
        print("⏳ Loading Whisper 'tiny' model on first use…")
        try:
            whisper_model = await asyncio.to_thread(_whisper_lib.load_model, "tiny")
            _whisper_model_loaded = True
            print("✅ Whisper 'tiny' model ready")
        except Exception as e:
            print(f"⚠  Whisper failed to load: {e}")

# ── Safe WebSocket send ───────────────────────────────────────
# FIXES the "send after close" crash. Every ws.send_json() call in this file
# goes through here so we never touch a dead socket.
async def safe_send(ws: WebSocket, payload: dict) -> bool:
    try:
        if ws.client_state != WebSocketState.CONNECTED:
            return False
        await ws.send_json(payload)
        return True
    except Exception:
        return False

# ── Remote Control Manager ───────────────────────────────────
# Tracks the single active desktop client and all phone remotes.
# Remote clients send commands; the manager relays them to the desktop.

class RemoteControlManager:
    def __init__(self):
        self.desktop: Optional[WebSocket] = None          # the main dashboard
        self.remotes: List[WebSocket]     = []            # phone controllers

    async def register_desktop(self, ws: WebSocket):
        self.desktop = ws
        print("📺 Desktop client registered for remote control")
        await safe_send(ws, {
            "type": "desktop_control_connected",
            "remotes": len(self.remotes),
        })

    def unregister_desktop(self, ws: WebSocket):
        if self.desktop is ws:
            self.desktop = None
            print("📺 Desktop client unregistered")

    async def register_remote(self, ws: WebSocket):
        self.remotes.append(ws)
        print(f"📱 Remote client connected  (total: {len(self.remotes)})")
        # Confirm connection to the phone
        await safe_send(ws, {"type": "remote_connected", "message": "Remote control active"})
        # Notify the desktop that a remote just joined
        if self.desktop:
            await safe_send(self.desktop, {
                "type":    "remote_joined",
                "remotes": len(self.remotes),
            })

    def unregister_remote(self, ws: WebSocket):
        if ws in self.remotes:
            self.remotes.remove(ws)
            print(f"📱 Remote client disconnected (total: {len(self.remotes)})")

    async def relay_to_desktop(self, payload: dict) -> bool:
        """Forward a remote action to the desktop client."""
        if not self.desktop:
            return False
        return await safe_send(self.desktop, payload)

    async def broadcast_to_remotes(self, payload: dict):
        """Push a state update from the desktop to all phones."""
        dead = []
        for r in list(self.remotes):
            ok = await safe_send(r, payload)
            if not ok:
                dead.append(r)
        for r in dead:
            self.unregister_remote(r)


remote_manager = RemoteControlManager()


# ── Remote WebSocket endpoint ─────────────────────────────────
@app.websocket("/ws/remote")
async def remote_ws(ws: WebSocket):
    """
    Phone remote-control endpoint.
    Each phone opens this connection, identifies itself as 'remote',
    then sends action messages that get relayed to the desktop.
    """
    await ws.accept()
    await remote_manager.register_remote(ws)
    try:
        while True:
            data = await ws.receive()
            if "text" not in data:
                continue
            try:
                msg = json.loads(data["text"])
                t   = msg.get("type")

                if t == "remote_action":
                    action = msg.get("action", "")
                    # Map remote actions → desktop message types
                    if action == "next":
                        ok = await remote_manager.relay_to_desktop({
                            "type":   "remote_next",
                            "source": "remote",
                        })
                        if not ok:
                            await safe_send(ws, {"type": "remote_action_status", "ok": False, "message": "Desktop control is not connected"})
                    elif action == "prev":
                        ok = await remote_manager.relay_to_desktop({
                            "type":   "remote_prev",
                            "source": "remote",
                        })
                        if not ok:
                            await safe_send(ws, {"type": "remote_action_status", "ok": False, "message": "Desktop control is not connected"})
                    elif action == "clear":
                        ok = await remote_manager.relay_to_desktop({
                            "type":   "remote_clear",
                            "source": "remote",
                        })
                        if not ok:
                            await safe_send(ws, {"type": "remote_action_status", "ok": False, "message": "Desktop control is not connected"})
                    elif action in {"start_transcription", "stop_transcription", "toggle_transcription"}:
                        ok = await remote_manager.relay_to_desktop({
                            "type":   "remote_" + action,
                            "source": "remote",
                        })
                        if not ok:
                            await safe_send(ws, {"type": "remote_action_status", "ok": False, "message": "Desktop control is not connected"})
                    elif action == "navigate":
                        ref = msg.get("ref", "")
                        if ref:
                            verse = bible.find_by_ref(ref)
                            payload = {
                                "type":   "remote_navigate",
                                "ref":    ref,
                                "source": "remote",
                            }
                            if verse:
                                payload["verse"] = verse
                                await safe_send(ws, {"type": "verse_state", "verse": verse, "translation": msg.get("translation", "KJV")})
                            ok = await remote_manager.relay_to_desktop(payload)
                            if not ok:
                                await safe_send(ws, {"type": "remote_action_status", "ok": False, "message": "Desktop control is not connected"})
                    elif action == "ping":
                        await safe_send(ws, {"type": "remote_pong"})

                elif t == "remote_ping":
                    await safe_send(ws, {"type": "remote_pong"})

            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        remote_manager.unregister_remote(ws)
        # Notify desktop that remote count changed
        if remote_manager.desktop:
            await safe_send(remote_manager.desktop, {
                "type":    "remote_left",
                "remotes": len(remote_manager.remotes),
            })


@app.websocket("/ws/control")
async def desktop_control_ws(ws: WebSocket):
    """
    Always-on desktop control channel.
    This stays alive as long as the desktop page is open, independent of the
    transcription/audio WebSocket, so phone commands can control the screen even
    when transcription is stopped or briefly paused.
    """
    await ws.accept()
    await remote_manager.register_desktop(ws)
    try:
        while True:
            data = await ws.receive()
            if "text" not in data or not data["text"]:
                continue
            try:
                msg = json.loads(data["text"])
            except json.JSONDecodeError:
                continue

            t = msg.get("type")
            if t == "desktop_state":
                # Desktop tells us the current verse — keep global_session in sync
                verse_data = msg.get("verse")
                if verse_data and verse_data.get("ref"):
                    global_session.set_current(verse_data)
                    bible.set_manual_index(verse_data["ref"])
                await remote_manager.broadcast_to_remotes({
                    "type": "verse_state",
                    "verse": msg.get("verse"),
                    "translation": msg.get("translation", "KJV"),
                    "transcribing": bool(msg.get("transcribing", False)),
                })
            elif t == "transcription_state":
                await remote_manager.broadcast_to_remotes({
                    "type": "transcription_state",
                    "active": bool(msg.get("active", False)),
                })
            elif t == "desktop_ping":
                await safe_send(ws, {"type": "desktop_pong", "remotes": len(remote_manager.remotes)})

            # ── Remote NEXT / PREV / NAVIGATE commands ────────────────
            # These arrive here (via relay_to_desktop) from the phone remote.
            # We handle them against global_session so they always act on the
            # last displayed verse, not an older transcription snapshot.
            elif t == "remote_next":
                if not global_session.current_ref:
                    await safe_send(ws, {"type": "remote_action_status", "ok": False, "message": "No verse loaded yet"})
                else:
                    verse = bible.get_adjacent_to_ref(global_session.current_ref, 1)
                    if verse:
                        global_session.set_current(verse)
                        bible.set_manual_index(verse["ref"])
                        add_to_feed(verse, "next_verse")
                        await safe_send(ws, {
                            "type": "next_verse",
                            "results": [verse],
                            "feed": list(scripture_feed),
                            "transcript_finalized": True,
                            "state": global_session.payload(),
                        })
                        await remote_manager.broadcast_to_remotes({
                            "type": "verse_state",
                            "verse": verse,
                            "translation": global_session.active_translation,
                        })
                    else:
                        await safe_send(ws, {"type": "remote_action_status", "ok": False, "message": "No next verse available"})

            elif t == "remote_prev":
                if not global_session.current_ref:
                    await safe_send(ws, {"type": "remote_action_status", "ok": False, "message": "No verse loaded yet"})
                else:
                    verse = bible.get_adjacent_to_ref(global_session.current_ref, -1)
                    if verse:
                        global_session.set_current(verse)
                        bible.set_manual_index(verse["ref"])
                        add_to_feed(verse, "previous_verse")
                        await safe_send(ws, {
                            "type": "exact_match",
                            "results": [verse],
                            "feed": list(scripture_feed),
                            "transcript_finalized": True,
                            "state": global_session.payload(),
                        })
                        await remote_manager.broadcast_to_remotes({
                            "type": "verse_state",
                            "verse": verse,
                            "translation": global_session.active_translation,
                        })
                    else:
                        await safe_send(ws, {"type": "remote_action_status", "ok": False, "message": "No previous verse available"})

            elif t == "remote_navigate":
                ref = msg.get("ref", "")
                verse = msg.get("verse") or (bible.find_by_ref(ref) if ref else None)
                if verse:
                    global_session.set_current(verse)
                    bible.set_manual_index(verse["ref"])
                    add_to_feed(verse, "navigate")
                    await safe_send(ws, {
                        "type": "exact_match",
                        "results": [verse],
                        "feed": list(scripture_feed),
                        "transcript_finalized": True,
                        "state": global_session.payload(),
                    })
                    await remote_manager.broadcast_to_remotes({
                        "type": "verse_state",
                        "verse": verse,
                        "translation": global_session.active_translation,
                    })
                else:
                    await safe_send(ws, {"type": "remote_action_status", "ok": False, "message": f"Verse not found: {ref}"})

            elif t == "remote_clear":
                await safe_send(ws, {"type": "clear_display"})
    except WebSocketDisconnect:
        pass
    finally:
        remote_manager.unregister_desktop(ws)


# ── Serve remote.html ─────────────────────────────────────────
def _find_remote_html() -> Optional[str]:
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "remote.html"),
        os.path.join(os.getcwd(), "remote.html"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

@app.get("/remote")
def serve_remote():
    path = _find_remote_html()
    if path:
        return FileResponse(path)
    return {"error": "remote.html not found — place it alongside main_fixed.py"}

def _get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"

@app.get("/remote-info")
def remote_info(request: Request):
    port = request.url.port or 8000
    lan_ip = _get_lan_ip()
    remote_port = LAN_PROXY_PORT_ACTIVE or port
    return {
        "lan_ip": lan_ip,
        "remote_port": remote_port,
        "remote_url": f"http://{lan_ip}:{remote_port}/remote",
        "desktop_url": f"http://127.0.0.1:{port}/",
    }


def _find_website_html() -> Optional[str]:
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "website.html"),
        os.path.join(os.getcwd(), "website.html"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

@app.get("/website")
def serve_website():
    path = _find_website_html()
    if path:
        return FileResponse(path)
    return {"error": "website.html not found — place it alongside main_fixed.py"}

@app.get("/download-package")
def download_package():
    base = os.path.dirname(os.path.abspath(__file__))
    exe_path = os.path.join(base, "start_in_the_beginning.exe")
    if os.path.exists(exe_path):
        return FileResponse(
            exe_path,
            media_type="application/octet-stream",
            filename="In The Beginning AI.exe"
        )
    # Fallback to zip if exe not found
    files = [
        ("main_fixed.py", os.path.join(base, "main_fixed.py")),
        ("In the Beginning.html", os.path.join(base, "In the Beginning.html")),
        ("remote.html", os.path.join(base, "remote.html")),
        ("website.html", os.path.join(base, "website.html")),
        ("start_in_the_beginning.py", os.path.join(base, "start_in_the_beginning.py")),
        ("Start In The Beginning.bat", os.path.join(base, "Start In The Beginning.bat")),
        ("requirements.txt", os.path.join(base, "requirements.txt")),
        (".env.example", os.path.join(base, ".env.example")),
    ]
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in files:
            if os.path.exists(path):
                zf.write(path, arcname)
    mem.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="in-the-beginning-software.zip"'}
    return StreamingResponse(mem, media_type="application/zip", headers=headers)


# ── Health ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":      "ok",
        "verse_count": len(bible.verses),
        "deepgram":    bool(SERVER_DG_KEY),
        "whisper":     WHISPER_AVAILABLE,
        "openai":      False,
        "vector":      bible.vector_status(),
        "mode":        "deepgram" if SERVER_DG_KEY else ("whisper" if WHISPER_AVAILABLE else "text-only"),
    }

# ── REST ──────────────────────────────────────────────────────
class SearchReq(BaseModel):
    text: str

@app.post("/search")
async def search(req: SearchReq):
    if not req.text.strip(): return {"results": [], "query": ""}
    exact = exact_lookup(req.text.strip())
    if exact: return {"results": [exact], "query": req.text}
    results = await asyncio.to_thread(bible.vector_search, req.text, TOP_K)
    if results: add_to_feed(results[0], "match")
    return {"results": results, "query": req.text}

@app.get("/lookup")
async def lookup(ref: str):
    result = exact_lookup(ref.strip())
    if result: return result
    return {"error": f"Verse not found: {ref}", "ref": ref}

@app.get("/feed")
def get_feed(): return {"feed": list(scripture_feed)}

@app.delete("/feed")
def clear_feed():
    scripture_feed.clear()
    return {"status": "cleared"}

# ── Display broadcast WebSocket ──────────────────────────────
# External projector windows (HDMI popup or OBS/NDI browser source) connect
# here.  The desktop frontend pushes verse payloads via POST /display/push,
# and every connected projector client receives them instantly without the
# user having to navigate back to the broadcast tab.
#
# Design: "framework-agnostic overlay" — the projector window is a plain
# <div> overlay (not fullscreen-forced) so OBS window-capture, NDI Tools
# browser source, ProPresenter Stage Display, or any external software can
# grab it at any size without it taking over the operator's screen.

_display_clients: List[WebSocket] = []

@app.websocket("/ws/display")
async def display_ws(ws: WebSocket):
    """Projector / NDI browser-source connects here for live verse pushes."""
    await ws.accept()
    _display_clients.append(ws)
    try:
        while True:
            # Keep alive; projector only receives, never sends
            try:
                await asyncio.wait_for(ws.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                await safe_send(ws, {"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ws in _display_clients:
            _display_clients.remove(ws)


async def _broadcast_to_display(payload: dict):
    """Push a verse to every connected projector/NDI client."""
    dead = []
    for client in list(_display_clients):
        ok = await safe_send(client, payload)
        if not ok:
            dead.append(client)
    for d in dead:
        if d in _display_clients:
            _display_clients.remove(d)


class DisplayPushReq(BaseModel):
    verse: Dict
    translation: Optional[str] = "KJV"
    theme: Optional[str] = "dark-blue"
    bg: Optional[str] = "#000011"


@app.post("/display/push")
async def display_push(req: DisplayPushReq):
    """
    The desktop frontend calls this whenever a verse changes.
    All connected /ws/display clients (projector windows, OBS browser sources)
    receive the verse immediately — no operator tab-switching required.
    """
    payload = {
        "type":        "verse",
        "verse":       req.verse,
        "translation": req.translation,
        "theme":       req.theme,
        "bg":          req.bg,
    }
    await _broadcast_to_display(payload)
    return {"pushed": True, "clients": len(_display_clients)}


@app.get("/display/status")
def display_status():
    return {"connected_clients": len(_display_clients)}


@app.get("/display")
def serve_display_page():
    """
    Standalone projector page — open in a browser window on the projector
    screen, OBS browser source, or NDI Tools virtual input.  It auto-connects
    to /ws/display and updates whenever a verse is pushed.  The overlay is
    anchored to the BOTTOM of the screen with a translucent pill so it never
    covers the full screen.
    """
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>In The Beginning · Display</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{
  background:transparent;
  min-height:100vh;
  font-family:'Georgia',serif;
  overflow:hidden;
  transition:background 0.5s ease;
}
#outer{
  position:fixed;
  left:50%;
  bottom:5vh;
  transform:translateX(-50%);
  max-width:1200px;
  width:88%;
  padding:22px 32px 22px 36px;
  text-align:center;
  background:rgba(0,0,0,0.72);
  border-left:5px solid #a67c52;
  border-radius:10px;
  box-shadow:0 16px 56px rgba(0,0,0,0.45);
  transition:opacity 0.4s ease,transform 0.4s ease;
}
#outer.empty{opacity:0;transform:translateX(-50%) translateY(20px);}
.ref{
  font-size:16px;color:#a67c52;font-weight:700;
  margin-bottom:10px;letter-spacing:1px;text-transform:uppercase;
}
.trans{color:#f9cb42;font-size:.78em;}
.txt{
  font-size:clamp(22px,2.8vw,40px);
  line-height:1.3;
  font-weight:400;
  color:#fff;
  display:-webkit-box;
  -webkit-line-clamp:4;
  -webkit-box-orient:vertical;
  overflow:hidden;
}
#badge{
  position:fixed;top:12px;right:16px;
  font-family:sans-serif;font-size:10px;font-weight:700;
  letter-spacing:2px;color:rgba(255,255,255,0.18);
  text-transform:uppercase;
}
#hint{
  position:fixed;bottom:10px;left:0;right:0;
  text-align:center;font-family:sans-serif;
  font-size:10px;opacity:0.2;color:#fff;
}
</style>
</head>
<body>
<div id="outer" class="empty">
  <div class="ref" id="vref"></div>
  <div class="txt" id="vtxt">Waiting for verse…</div>
</div>
<div id="badge">In The Beginning · Live</div>
<div id="hint">F = fullscreen &nbsp;·&nbsp; Esc = exit</div>
<script>
(function(){
  var outer = document.getElementById('outer');
  var vref  = document.getElementById('vref');
  var vtxt  = document.getElementById('vtxt');
  var proto = location.protocol === 'https:' ? 'wss' : 'ws';
  var wsUrl = proto + '://' + location.host + '/ws/display';
  var ws, retryDelay = 1500;

  function connect(){
    ws = new WebSocket(wsUrl);
    ws.onmessage = function(e){
      try{
        var d = JSON.parse(e.data);
        if(d.type === 'verse' && d.verse){
          var v = d.verse;
          vref.innerHTML = (v.ref||'') +
            ' <span class="trans">[' + (v.translation||d.translation||'KJV') + ']</span>';
          vtxt.textContent = v.text || '';
          if(d.bg) document.body.style.background = d.bg;
          outer.classList.remove('empty');
        }
      }catch(ex){}
    };
    ws.onclose = function(){
      setTimeout(connect, retryDelay);
      retryDelay = Math.min(retryDelay * 1.5, 10000);
    };
    ws.onopen = function(){ retryDelay = 1500; };
  }
  connect();

  document.addEventListener('keydown',function(e){
    if(e.key==='F'||e.key==='f'){
      if(!document.fullscreenElement) document.documentElement.requestFullscreen();
      else document.exitFullscreen();
    }
  });
})();
</script>
</body>
</html>"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(html)


@app.get("/chapter")
def get_chapter(book: str, chapter: int):
    book = canonicalize_book_name(book)
    prefix = f"{book} {chapter}:".lower()
    verses = [v for v in bible.verses if v["ref"].lower().startswith(prefix)]
    return {"book": book, "chapter": chapter, "verses": verses}

class LiveSearchReq(BaseModel):
    text: str

class TranslationImportReq(BaseModel):
    code: str
    content: str
    format: Optional[str] = None
    source_name: Optional[str] = None

class TranslationApiReq(BaseModel):
    code: str
    url_template: str
    response_path: Optional[str] = None
    api_key: Optional[str] = None
    header_name: Optional[str] = "Authorization"

@app.post("/live-search")
async def live_search(req: LiveSearchReq):
    text = req.text.strip()
    if not text or len(text) < 3:
        return {"results": [], "is_exact": False, "query": text}
    clean = normalise_query(text)
    exact = exact_lookup(clean)
    if exact:
        return {"results": [exact], "is_exact": True, "query": text}
    results = await asyncio.to_thread(bible.vector_search, clean, TOP_K)
    return {"results": results, "is_exact": False, "query": text}

# ── Helpers ───────────────────────────────────────────────────
NUMBER_WORDS = {
    "zero": 0, "oh": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

def _normalise_intent_text(text: str) -> str:
    clean = text.lower().replace("-", " ")
    clean = re.sub(r"[^\w\s]", " ", clean)
    return re.sub(r"\s+", " ", clean).strip()

def _consume_number(tokens: List[str], start: int) -> Tuple[Optional[int], int]:
    if start >= len(tokens):
        return None, start
    token = tokens[start]
    if re.fullmatch(r"\d+", token):
        return int(token), start + 1

    total = 0
    i = start
    found = False
    while i < len(tokens):
        token = tokens[i]
        if token == "and":
            i += 1
            continue
        if token not in NUMBER_WORDS:
            break
        total += NUMBER_WORDS[token]
        found = True
        i += 1
        if token not in {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}:
            break
    return (total if found and total > 0 else None), i

def parse_reference_from_text(text: str) -> Optional[str]:
    clean = _normalise_intent_text(text)
    clean = re.sub(r"\b(?:open|lookup|look up|read|show|display|pull up|go to|turn to|find|scripture)\b", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    for alias, book in bible.book_aliases():
        match = re.search(rf"(?:^|\b){re.escape(alias)}(?:\b|$)(?P<tail>.*)$", clean)
        if not match:
            continue
        tokens = match.group("tail").strip().split()
        i = 0
        while i < len(tokens) and tokens[i] in {"chapter", "chap", "ch"}:
            i += 1
        chapter, i = _consume_number(tokens, i)
        if chapter is None:
            continue
        while i < len(tokens) and tokens[i] in {"and", "verse", "verses", "vs", "v", "number"}:
            i += 1
        verse, _ = _consume_number(tokens, i)
        if verse is None:
            continue
        return f"{book} {chapter}:{verse}"
    return None

def parse_reference_parts_from_text(text: str) -> Optional[Dict]:
    clean = _normalise_intent_text(text)
    clean = re.sub(r"\b(?:open|lookup|look up|read|show|display|pull up|go to|turn to|find|scripture)\b", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return None

    for alias, book in bible.book_aliases():
        match = re.search(rf"(?:^|\b){re.escape(alias)}(?:\b|$)(?P<tail>.*)$", clean)
        if not match:
            continue
        tokens = match.group("tail").strip().split()
        i = 0
        while i < len(tokens) and tokens[i] in {"chapter", "chap", "ch"}:
            i += 1
        chapter, i = _consume_number(tokens, i)
        if chapter is None:
            return {"book": book, "chapter": None, "verse": None, "complete": False}
        while i < len(tokens) and tokens[i] in {"and", "verse", "verses", "vs", "v", "number"}:
            i += 1
        verse, _ = _consume_number(tokens, i)
        return {"book": book, "chapter": chapter, "verse": verse, "complete": verse is not None}
    return None

def exact_lookup(ref: str) -> Optional[Dict]:
    """Match a reference string against the Bible index — two passes."""
    parsed_ref = parse_reference_from_text(ref)
    if parsed_ref:
        indexed = bible.find_by_ref(parsed_ref)
        if indexed:
            return indexed
    indexed = bible.find_by_ref(ref)
    if indexed:
        return indexed
    candidates = []
    for candidate in (ref, normalise_reference_text(ref)):
        normalized = re.sub(r"\s+", " ", candidate.lower()).strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    for candidate in candidates:
        for v in bible.verses:
            if v["ref"].lower().strip() == candidate:
                return {**v, "score": 1.0}

    for candidate in candidates:
        candidate_nospace = candidate.replace(" ", "")
        for v in bible.verses:
            if v["ref"].lower().replace(" ", "") == candidate_nospace:
                return {**v, "score": 1.0}
    return None

def normalise_query(query: str) -> str:
    """
    Translate spoken words → searchable numbers/symbols.
    "First Corinthians ten verse twelve" → "1 corinthians 10:12"
    """
    parsed_ref = parse_reference_from_text(query)
    if parsed_ref:
        return parsed_ref
    clean = query.lower()
    num_map = {
        'first': '1', 'second': '2', 'third': '3', 'one': '1', 'two': '2',
        'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7',
        'eight': '8', 'nine': '9', 'ten': '10', 'eleven': '11', 'twelve': '12',
        'thirteen': '13', 'fourteen': '14', 'fifteen': '15', 'sixteen': '16',
        'seventeen': '17', 'eighteen': '18', 'nineteen': '19',
        'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50',
        # ordinals
        '1st': '1', '2nd': '2', '3rd': '3',
        'verse': ':', 'chapter': ' ',
    }
    for word, replacement in num_map.items():
        clean = re.sub(rf'\b{re.escape(word)}\b', replacement, clean)
    clean = re.sub(r'(\d+)\s*(?:to|-|and)\s*\d+', r'\1', clean)   # strip ranges
    clean = re.sub(r'((?:[1-3]\s+)?[a-z]+)\s+(\d+)\s+(\d+)', r'\1 \2:\3', clean)
    clean = re.sub(r'\s*:\s*', ':', clean).strip()
    return clean

# ── Semantic Topic Expander ───────────────────────────────────
# Maps church-preaching themes to enriched embedding queries so that
# FAISS can surface topically relevant verses even when the preacher's
# exact words don't appear in scripture text.
# Each entry: (regex pattern, expanded query for BGE embedding)
_TOPIC_EXPANSIONS: List[Tuple[re.Pattern, str]] = [
    # Soul winning / evangelism
    (re.compile(r"\b(soul[s]?\s*win|win(?:ning)?\s*soul|lead\s*(?:someone|them|people|sinners?)\s*to\s*(?:christ|god|salvation|jesus)|evangelis[mt]|reach\s*the\s*lost|go\s*and\s*preach|great\s*commission|tell\s*the\s*world|harvest\s*of\s*soul)\b", re.I),
     "value of one soul salvation rejoice heaven lost sheep found sinner repents God joy"),

    # Holy Spirit teaching / helper
    (re.compile(r"\b(holy\s*spirit\s*(?:teach|will\s*teach|guide|instruct|remind|counsel)|spirit\s*(?:of\s*truth|teach|guide|comfort)|comforter|paraclete|spirit\s*lead)\b", re.I),
     "Holy Spirit teach all things bring to remembrance Comforter guide truth"),

    # Faith / believing
    (re.compile(r"\b(faith\s*(?:in\s*god|move[s]?\s*mountain|without\s*doubt|heal|miracle)|walk\s*by\s*faith|believe\s*and\s*(?:receive|it\s*shall)|trust\s*(?:in\s*the\s*lord|god))\b", re.I),
     "faith trust believe God mountain doubt nothing impossible"),

    # Prayer / asking God
    (re.compile(r"\b(pow(?:er)?\s*of\s*prayer|pray(?:ing)?\s*without\s*ceas|ask\s*and\s*(?:ye\s*shall|you\s*will)\s*receive|seek\s*and\s*(?:ye\s*shall|you\s*will)\s*find|knock\s*and\s*(?:the\s*door|it\s*shall)|effectual\s*fervent\s*prayer)\b", re.I),
     "ask seek knock prayer receive answered faith persistent"),

    # Grace / salvation by grace
    (re.compile(r"\b(saved\s*by\s*grace|grace\s*(?:of\s*god|through\s*faith|not\s*works)|gift\s*of\s*(?:god|salvation)|not\s*by\s*work[s]?|unmerited\s*favor)\b", re.I),
     "grace saved through faith not works gift God righteousness"),

    # Love of God / God's love
    (re.compile(r"\b(god[']?s?\s*love|love\s*of\s*(?:god|christ|the\s*father)|god\s*so\s*loved|unconditional\s*love|father[']?s?\s*love|loved\s*(?:us|the\s*world)\s*(?:so\s*much|first))\b", re.I),
     "God so loved the world gave only begotten Son everlasting love"),

    # Healing / divine healing
    (re.compile(r"\b(divine\s*heal|god\s*(?:heal[s]?|is\s*(?:a\s*)?healer)|by\s*(?:his|whose)\s*stripe[s]?|stripes\s*(?:we\s*are|ye\s*were)\s*heal|healing\s*(?:power|virtue|anointing))\b", re.I),
     "healed stripes wounds sick recover lay hands healing"),

    # Anointing / power of God
    (re.compile(r"\b(anointing\s*(?:of\s*god|break[s]?\s*yoke|fall[s]?\s*on|upon\s*me)|yoke\s*(?:destroying|breaking)\s*anointing|power\s*from\s*on\s*high|baptis[em]\s*(?:of\s*)?(?:the\s*)?holy\s*spirit|power\s*of\s*god)\b", re.I),
     "anointing oil yoke broken power Holy Ghost upon me Spirit"),

    # Prosperity / blessing
    (re.compile(r"\b(god[']?s?\s*(?:provision|supply|prosper)|prosper(?:ity|ous)?|bless(?:ing|ed|ings)?\s*(?:of\s*god|of\s*the\s*lord|overflow)|abundance\s*(?:of\s*god|life)|all\s*(?:your|my)\s*need[s]?\s*(?:met|supplied))\b", re.I),
     "prosper abundance bless supply needs met give good gifts"),

    # Second coming / rapture / end times
    (re.compile(r"\b(second\s*com(?:ing)?|rapture|caught\s*up|trump(?:et)?\s*of\s*god|lord\s*(?:shall\s*)?descend|dead\s*in\s*christ|end\s*times|last\s*days|parousia)\b", re.I),
     "Lord descend shout trumpet dead Christ rise caught up clouds"),

    # Redemption / blood of Jesus
    (re.compile(r"\b(blood\s*of\s*(?:jesus|christ|the\s*lamb)|redeem(?:ed|ption|ing)?|lamb\s*of\s*god|atonement|ransom|bought\s*(?:with\s*a\s*price|by\s*(?:his|the)\s*blood)|precious\s*blood)\b", re.I),
     "blood Jesus Christ redeemed forgiven atonement lamb sacrifice sin"),

    # Word of God / scripture
    (re.compile(r"\b(word\s*of\s*god|scripture\s*(?:says|tells|is)|bible\s*says|word\s*is\s*(?:a\s*lamp|alive|sharper|powerful)|rhema|logos|thy\s*word)\b", re.I),
     "word God lamp light path sword scripture profitable doctrine"),

    # Worship / praise
    (re.compile(r"\b(worship\s*(?:god|in\s*spirit|in\s*truth)|praise\s*(?:the\s*lord|god|his\s*name)|enter\s*(?:his\s*)?gate[s]?\s*with|sacrifice\s*of\s*praise|shout\s*unto\s*the\s*lord|hallelujah)\b", re.I),
     "praise worship Lord shout joy enter gates thanksgiving holy"),

    # Fear not / courage
    (re.compile(r"\b(fear\s*not|do\s*not\s*be\s*afraid|be\s*(?:strong\s*and\s*courageous|not\s*dismayed)|god\s*(?:is\s*with\s*you|has\s*not\s*given\s*us\s*a\s*spirit\s*of\s*fear))\b", re.I),
     "fear not afraid strong courageous God with you spirit power love sound mind"),

    # Forgiveness / sin
    (re.compile(r"\b(forgiv(?:e|en|eness|ing)|confess\s*(?:sin[s]?|our\s*sin)|remission\s*of\s*sin|washed\s*(?:clean|white)|repent(?:ance)?|turn\s*from\s*sin)\b", re.I),
     "forgive confess sin cleanse repent remission blood righteous"),

    # Heaven / eternal life
    (re.compile(r"\b(eternal\s*(?:life|home)|heaven(?:ly\s*father)?|mansions?\s*(?:in\s*heaven|prepared)|kingdom\s*(?:of\s*heaven|of\s*god)|everlasting\s*life|life\s*after\s*death|paradise)\b", re.I),
     "eternal life heaven mansions prepared believe not perish everlasting"),

    # Armor of God / spiritual warfare
    (re.compile(r"\b(armor\s*of\s*god|spiritual\s*(?:warfare|battle|weapon[s]?)|wrestle\s*not\s*against\s*flesh|put\s*on\s*(?:the\s*)?(?:full\s*)?armor|sword\s*of\s*the\s*spirit|shield\s*of\s*faith)\b", re.I),
     "armor God spiritual warfare principalities sword faith shield righteousness"),

    # Peace / rest in God
    (re.compile(r"\b(peace\s*(?:of\s*god|that\s*passes|that\s*surpasses|be\s*still)|rest\s*in\s*(?:god|the\s*lord)|cast\s*(?:all\s*)?(?:your\s*)?(?:anxiety|care|burden[s]?)\s*on|be\s*anxious\s*for\s*nothing)\b", re.I),
     "peace God surpasses understanding anxious worry cast care still know"),

    # Strength / weakness made strong
    (re.compile(r"\b(strength(?:en)?\s*(?:in\s*(?:the\s*)?lord|through\s*christ)|i\s*can\s*do\s*all\s*things|made\s*strong\s*in\s*weakness|mount\s*up\s*(?:with\s*)?wings|renew\s*(?:ed)?\s*strength|wait\s*on\s*(?:the\s*)?lord)\b", re.I),
     "strength weak Christ strengthens all things possible wings eagles renew"),
]

def _expand_semantic_query(query: str) -> str:
    """
    Check if the query matches a known preaching theme and return an
    enriched embedding query that helps FAISS surface topically relevant
    verses.  Falls back to the original query if no theme matches.
    """
    for pattern, expansion in _TOPIC_EXPANSIONS:
        if pattern.search(query):
            # Blend the original query with the theme expansion so that
            # highly specific phrasing still anchors the search while
            # thematic coverage is broadened.
            return f"{query} {expansion}"
    return query


class TranslationSourceRegistry:
    """
    Legal-safe Bible source layer:
    - KJV comes from the public-domain local index.
    - User imports/API connectors are runtime sources supplied by the user.
    - Imported/API content stays in memory and is not bundled with the app.
    """
    def __init__(self):
        self.imported: Dict[str, Dict[str, str]] = {}
        self.api_sources: Dict[str, Dict[str, Any]] = {}
        self.source_meta: Dict[str, Dict[str, Any]] = {"KJV": {"type": "public_domain", "count": 0}}

    def _code(self, code: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9_-]", "", (code or "").upper())
        if not clean:
            raise ValueError("Translation code is required.")
        return clean[:16]

    def _ref_key(self, ref: str) -> str:
        return normalise_reference_text(ref).lower().strip()

    def _put(self, verses: Dict[str, str], ref: str, text: str):
        clean_text = re.sub(r"\s+", " ", (text or "").strip())
        if not clean_text:
            return
        normalized = normalise_reference_text(ref)
        if re.search(r"\d+:\d+$", normalized):
            verses[self._ref_key(normalized)] = clean_text

    def _parse_json(self, content: str) -> Dict[str, str]:
        data = json.loads(content)
        verses: Dict[str, str] = {}

        if isinstance(data, dict) and all(isinstance(k, str) and ":" in k for k in list(data)[:5]):
            for ref, text in data.items():
                self._put(verses, ref, str(text))
            return verses

        if isinstance(data, dict) and isinstance(data.get("verses"), list):
            for item in data["verses"]:
                if not isinstance(item, dict):
                    continue
                ref = item.get("ref") or item.get("reference")
                if not ref and all(k in item for k in ("book", "chapter", "verse")):
                    ref = f"{item['book']} {item['chapter']}:{item['verse']}"
                self._put(verses, str(ref or ""), str(item.get("text", "")))
            return verses

        if isinstance(data, dict):
            for book, chapters in data.items():
                if isinstance(chapters, dict):
                    iterable = chapters.items()
                elif isinstance(chapters, list):
                    iterable = enumerate(chapters, start=1)
                else:
                    continue
                for chapter_no, chapter in iterable:
                    if isinstance(chapter, dict):
                        verse_iter = chapter.items()
                    elif isinstance(chapter, list):
                        verse_iter = enumerate(chapter, start=1)
                    else:
                        continue
                    for verse_no, text in verse_iter:
                        if isinstance(text, dict):
                            text = text.get("text", "")
                        self._put(verses, f"{book} {chapter_no}:{verse_no}", str(text))
        return verses

    def _parse_txt(self, content: str) -> Dict[str, str]:
        verses: Dict[str, str] = {}
        for line in content.splitlines():
            clean = line.strip()
            if not clean:
                continue
            match = re.match(r"^(.+?\s+\d+\s*:\s*\d+)\s+(.+)$", clean)
            if match:
                self._put(verses, match.group(1), match.group(2))
        return verses

    def _parse_xml(self, content: str) -> Dict[str, str]:
        verses: Dict[str, str] = {}
        root = ET.fromstring(content)
        for el in root.iter():
            attrs = {k.lower(): v for k, v in el.attrib.items()}
            ref = attrs.get("ref") or attrs.get("reference") or attrs.get("osisid")
            if not ref and all(k in attrs for k in ("book", "chapter", "verse")):
                ref = f"{attrs['book']} {attrs['chapter']}:{attrs['verse']}"
            text = " ".join(t.strip() for t in el.itertext() if t and t.strip())
            if ref and text:
                xml_ref = ref.replace(".", " ")
                osis_match = re.match(r"^(.+?)\s+(\d+)\s+(\d+)$", xml_ref)
                if osis_match:
                    xml_ref = f"{osis_match.group(1)} {osis_match.group(2)}:{osis_match.group(3)}"
                self._put(verses, xml_ref, text)
        return verses

    def import_content(self, req: TranslationImportReq) -> Dict:
        code = self._code(req.code)
        fmt = (req.format or "").lower().strip()
        content = req.content or ""
        if not fmt:
            start = content.lstrip()[:1]
            fmt = "json" if start in ("{", "[") else ("xml" if start == "<" else "txt")

        if fmt == "json":
            verses = self._parse_json(content)
        elif fmt == "xml":
            verses = self._parse_xml(content)
        elif fmt in {"txt", "text"}:
            verses = self._parse_txt(content)
        else:
            raise ValueError("Supported formats are json, xml, and txt.")

        if not verses:
            raise ValueError("No verses were found. Use refs like 'John 3:16' with verse text.")

        self.imported[code] = verses
        self.source_meta[code] = {
            "type": "user_import",
            "count": len(verses),
            "source_name": req.source_name or code,
        }
        return {"code": code, "count": len(verses), "type": "user_import"}

    def connect_api(self, req: TranslationApiReq) -> Dict:
        code = self._code(req.code)
        template = (req.url_template or "").strip()
        if not template.startswith(("https://", "http://")) or "{ref}" not in template:
            raise ValueError("API URL template must start with http(s) and include {ref}.")
        self.api_sources[code] = {
            "url_template": template,
            "response_path": (req.response_path or "").strip(),
            "api_key": (req.api_key or "").strip(),
            "header_name": (req.header_name or "Authorization").strip(),
        }
        self.source_meta[code] = {"type": "licensed_api", "count": None}
        return {"code": code, "type": "licensed_api"}

    def lookup_imported(self, code: str, ref: str) -> Optional[str]:
        source = self.imported.get(self._code(code))
        if not source:
            return None
        return source.get(self._ref_key(ref))

    def _dig(self, data: Any, path: str) -> Any:
        current = data
        for part in [p for p in path.split(".") if p]:
            if isinstance(current, list) and part.isdigit():
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    async def lookup(self, code: str, ref: str) -> Optional[Dict]:
        code = self._code(code)
        imported = self.lookup_imported(code, ref)
        if imported:
            return {"code": code, "ref": normalise_reference_text(ref), "text": imported, "source": "user_import"}

        api = self.api_sources.get(code)
        if not api:
            return None

        parsed = bible.split_ref(normalise_reference_text(ref))
        book, chapter, verse = parsed if parsed else ("", "", "")
        url = api["url_template"].format(
            ref=quote(normalise_reference_text(ref)),
            book=quote(str(book)),
            chapter=quote(str(chapter)),
            verse=quote(str(verse)),
            translation=quote(code),
        )
        headers = {}
        if api.get("api_key"):
            header_name = api.get("header_name") or "Authorization"
            headers[header_name] = api["api_key"]

        import httpx
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "json" in content_type:
                data = resp.json()
                value = self._dig(data, api.get("response_path") or "") if api.get("response_path") else None
                if value is None and isinstance(data, dict):
                    value = data.get("text") or data.get("verse") or data.get("content")
            else:
                value = resp.text

        text = re.sub(r"\s+", " ", str(value or "").strip())
        return {"code": code, "ref": normalise_reference_text(ref), "text": text, "source": "licensed_api"} if text else None

    def list_sources(self) -> List[Dict]:
        self.source_meta["KJV"]["count"] = len(bible.verses)
        return [{"code": code, **meta} for code, meta in sorted(self.source_meta.items())]

translation_sources = TranslationSourceRegistry()

@app.get("/translations")
def list_translations():
    return {"sources": translation_sources.list_sources()}

@app.post("/translations/import")
async def import_translation(req: TranslationImportReq):
    try:
        return translation_sources.import_content(req)
    except Exception as e:
        return {"error": str(e)}

@app.post("/translations/connect")
async def connect_translation_api(req: TranslationApiReq):
    try:
        return translation_sources.connect_api(req)
    except Exception as e:
        return {"error": str(e)}

@app.get("/translations/lookup")
async def lookup_translation(code: str, ref: str):
    if code.upper() == "KJV":
        verse = exact_lookup(ref)
        if verse:
            return {"code": "KJV", "ref": verse["ref"], "text": verse["text"], "source": "public_domain"}
    try:
        result = await translation_sources.lookup(code, ref)
        if result:
            return result
        return {"error": f"No source configured for {code} or verse not found.", "code": code, "ref": ref}
    except Exception as e:
        return {"error": str(e), "code": code, "ref": ref}

@dataclass
class SessionState:
    current_book: Optional[str] = None
    current_chapter: Optional[int] = None
    current_verse: Optional[int] = None
    current_ref: Optional[str] = None
    pending_book: Optional[str] = None
    pending_chapter: Optional[int] = None
    pending_updated_at: float = 0.0
    active_translation: str = "KJV"
    last_command_signature: str = ""
    last_command_at: float = 0.0

    def set_current(self, verse: Dict):
        parsed = bible.split_ref(verse.get("ref", ""))
        if not parsed:
            return
        book, chapter, verse_no = parsed
        self.current_book = book
        self.current_chapter = chapter
        self.current_verse = verse_no
        self.current_ref = f"{book} {chapter}:{verse_no}"
        self.clear_pending_reference()

    def set_pending_reference(self, book: str, chapter: Optional[int] = None):
        self.pending_book = canonicalize_book_name(book)
        self.pending_chapter = chapter
        self.pending_updated_at = time.monotonic()

    def clear_pending_reference(self):
        self.pending_book = None
        self.pending_chapter = None
        self.pending_updated_at = 0.0

    def pending_reference_active(self, timeout_sec: float = 20.0) -> bool:
        if not self.pending_book:
            return False
        if time.monotonic() - self.pending_updated_at > timeout_sec:
            self.clear_pending_reference()
            return False
        return True

    def payload(self) -> Dict:
        return {
            "current_book": self.current_book,
            "current_chapter": self.current_chapter,
            "current_verse": self.current_verse,
            "current_ref": self.current_ref,
            "pending_book": self.pending_book,
            "pending_chapter": self.pending_chapter,
            "active_translation": self.active_translation,
        }

# Initialise the shared session now that SessionState is defined
global_session = SessionState()

def _parse_spoken_int(value: str) -> Optional[int]:
    clean = _normalise_intent_text(value)
    digit = re.search(r"\d+", clean)
    if digit:
        return int(digit.group(0))

    total = 0
    found = False
    for token in clean.split():
        if token in {"and", "verse", "number"}:
            continue
        if token not in NUMBER_WORDS:
            break
        total += NUMBER_WORDS[token]
        found = True
    return total if found and total > 0 else None

def _extract_spoken_numbers(value: str, limit: int = 2) -> List[int]:
    tokens = _normalise_intent_text(value).split()
    out: List[int] = []
    i = 0
    while i < len(tokens) and len(out) < limit:
        if tokens[i] in {"chapter", "chap", "ch", "verse", "verses", "vs", "v", "number", "and"}:
            i += 1
            continue
        number, next_i = _consume_number(tokens, i)
        if number is not None:
            out.append(number)
            i = max(next_i, i + 1)
        else:
            i += 1
    return out

def resolve_direct_reference_lane(query: str, session: SessionState) -> Tuple[str, Optional[Dict], Optional[str]]:
    """
    Lane 1: direct catcher. Returns:
      ("exact", verse, None)    complete citation found
      ("pending", None, text)   partial citation remembered
      ("none", None, None)      let semantic lane handle it
    """
    parts = parse_reference_parts_from_text(query)
    if parts:
        book = parts["book"]
        chapter = parts.get("chapter")
        verse_no = parts.get("verse")
        if parts.get("complete"):
            session.clear_pending_reference()
            verse = bible.find_by_parts(book, int(chapter), int(verse_no))
            if verse:
                return "exact", verse, None
        session.set_pending_reference(book, chapter)
        if chapter:
            return "pending", None, f"Heard {book} {chapter}; waiting for verse."
        return "pending", None, f"Heard {book}; waiting for chapter and verse."

    if session.pending_reference_active():
        clean = _normalise_intent_text(query)
        if not re.search(r"\b(chapter|chap|ch|verse|verses|vs|v|number|\d|zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b", clean):
            return "none", None, None
        numbers = _extract_spoken_numbers(clean, limit=2)
        if not numbers:
            return "pending", None, f"Heard {session.pending_book}; waiting for chapter and verse."

        if session.pending_chapter is None:
            chapter = numbers[0]
            verse_no = numbers[1] if len(numbers) > 1 else None
        else:
            chapter = session.pending_chapter
            verse_no = numbers[0]

        if verse_no is not None:
            pending_book = session.pending_book or ""
            verse = bible.find_by_parts(pending_book, int(chapter), int(verse_no))
            session.clear_pending_reference()
            if verse:
                return "exact", verse, None
            return "pending", None, f"Could not find {pending_book} {chapter}:{verse_no}."

        session.set_pending_reference(session.pending_book or "", int(chapter))
        return "pending", None, f"Heard {session.pending_book} {chapter}; waiting for verse."

    return "none", None, None

def _translation_from_phrase(value: str) -> Optional[str]:
    clean = _normalise_intent_text(value)
    clean = re.sub(r"\b(the|a|an|bible|version|translation|please)\b", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return None
    if clean in TRANSLATION_ALIASES:
        return TRANSLATION_ALIASES[clean]
    for alias, code in sorted(TRANSLATION_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", clean):
            return code
    upper = clean.upper()
    return upper if re.fullmatch(r"[A-Z0-9]{2,8}", upper) else None

def parse_voice_intent(query: str, session: SessionState) -> Optional[Dict]:
    clean = _normalise_intent_text(query)
    if not clean:
        return None

    for pattern in (
        r"\b(?:give|show|display|put)\s+(?:me\s+|us\s+|it\s+|this\s+|the\s+verse\s+)?in\s+(?P<name>[a-z0-9\s]+)$",
        r"\b(?:switch|change|set)\s+(?:the\s+)?(?:translation\s+|version\s+)?(?:to|into)\s+(?P<name>[a-z0-9\s]+)$",
        r"\b(?P<name>[a-z0-9\s]+)\s+(?:version|translation)$",
    ):
        match = re.search(pattern, clean)
        if match:
            translation = _translation_from_phrase(match.group("name"))
            if translation:
                return {"kind": "translation", "action": "set_translation", "translation": translation}

    show_match = re.search(
        r"\b(?:show|display|go to|jump to|take me to)\s+(?:me\s+)?verse\s+(?P<number>[a-z0-9\s-]+)$",
        clean,
    )
    if show_match:
        verse_no = _parse_spoken_int(show_match.group("number"))
        if verse_no:
            return {"kind": "navigation", "action": "show_verse", "verse": verse_no}

    if NEXT_VERSE_RE.search(clean):
        return {"kind": "navigation", "action": "next_verse"}
    if PREVIOUS_VERSE_RE.search(clean):
        return {"kind": "navigation", "action": "previous_verse"}

    return None

def _annotate_for_session(verse: Dict, session: SessionState) -> Dict:
    payload = {**verse, "translation": session.active_translation}
    if session.active_translation != "KJV":
        translated = translation_sources.lookup_imported(session.active_translation, verse.get("ref", ""))
        if translated:
            payload["text"] = translated
            payload["translation_source"] = "user_import"
        else:
            payload["translation_missing"] = True
    return payload

def _command_duplicate(session: SessionState, intent: Dict, query: str) -> bool:
    signature = json.dumps({
        "kind": intent.get("kind"),
        "action": intent.get("action"),
        "translation": intent.get("translation"),
        "verse": intent.get("verse"),
    }, sort_keys=True)
    now = time.monotonic()
    if session.last_command_signature == signature and now - session.last_command_at < 1.6:
        return True
    session.last_command_signature = signature
    session.last_command_at = now
    return False

async def _send_command_execution(
    ws: WebSocket,
    session: SessionState,
    query: str,
    intent: Dict,
    success: bool,
    message: str,
    verse: Optional[Dict] = None,
    feed: Optional[List[Dict]] = None,
    is_interim: bool = False,
    transcript_finalized: bool = False,
):
    payload = {
        "type": "command_execution",
        "command": intent.get("kind"),
        "action": intent.get("action"),
        "success": success,
        "message": message,
        "transcript": query,
        "state": session.payload(),
        "results": [_annotate_for_session(verse, session)] if verse else [],
        "feed": feed if feed is not None else list(scripture_feed),
        "is_interim": is_interim,
        "transcript_finalized": transcript_finalized,
    }
    await safe_send(ws, payload)

async def execute_voice_intent(
    ws: WebSocket,
    query: str,
    intent: Dict,
    session: SessionState,
    is_interim: bool = False,
    transcript_finalized: bool = False,
    dedupe: bool = True,
) -> bool:
    if dedupe and _command_duplicate(session, intent, query):
        return True

    action = intent.get("action")

    if action == "set_translation":
        session.active_translation = intent["translation"]
        verse = bible.find_by_ref(session.current_ref) if session.current_ref else None
        if verse:
            session.set_current(verse)
        await _send_command_execution(
            ws, session, query, intent, True,
            f"Translation switched to {session.active_translation}.",
            verse=verse,
            is_interim=is_interim,
            transcript_finalized=transcript_finalized,
        )
        return True

    if action in {"next_verse", "previous_verse", "show_verse"}:
        if not session.current_ref:
            await _send_command_execution(
                ws, session, query, intent, False,
                "No current verse is loaded yet.",
                is_interim=is_interim,
                transcript_finalized=transcript_finalized,
            )
            return True

        if action == "next_verse":
            verse = bible.get_adjacent_to_ref(session.current_ref, 1)
            label = "next"
        elif action == "previous_verse":
            verse = bible.get_adjacent_to_ref(session.current_ref, -1)
            label = "previous"
        else:
            verse = bible.find_by_parts(session.current_book or "", session.current_chapter or 0, intent["verse"])
            label = f"verse {intent['verse']}"

        if not verse:
            await _send_command_execution(
                ws, session, query, intent, False,
                f"Could not load {label} from the current chapter.",
                is_interim=is_interim,
                transcript_finalized=transcript_finalized,
            )
            return True

        session.set_current(verse)
        bible.set_manual_index(verse["ref"])
        annotated = _annotate_for_session(verse, session)
        add_to_feed(annotated, action)
        await _send_command_execution(
            ws, session, query, intent, True,
            f"Showing {annotated['ref']} in {session.active_translation}.",
            verse=verse,
            feed=list(scripture_feed),
            is_interim=is_interim,
            transcript_finalized=transcript_finalized,
        )
        return True

    return False

def _find_verse_by_ref(ref: Optional[str], candidates: List[Dict]) -> Optional[Dict]:
    if not ref: return None
    ref_l = ref.lower().strip()
    for v in candidates:
        if v["ref"].lower().strip() == ref_l:
            return v
    return None

def _canonical_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()

def _quick_verbatim_match(query: str, candidates: List[Dict]) -> Optional[Dict]:
    """
    Fast path for live verbatim scripture calls.
    We only treat it as exact if the spoken text is a meaningful contiguous
    phrase from the verse text, which keeps interim exact-matches precise.
    """
    canon_query = _canonical_text(query)
    query_words = canon_query.split()
    if len(query_words) < 4:
        return None

    for verse in candidates[:3]:
        canon_verse = _canonical_text(verse.get("text", ""))
        if canon_query and canon_query in canon_verse:
            return verse
    return None

def _is_last_feed_ref(ref: str) -> bool:
    return bool(scripture_feed and scripture_feed[-1].get("ref") == ref)

async def _send_exact_match(
    ws: WebSocket,
    query: str,
    verse: Dict,
    session: SessionState,
    confidence: float = 1.0,
    openai: bool = False,
    transcript_finalized: bool = False,
    is_interim: bool = False,
    payload_type: str = "exact_match",
    feed_type: str = "exact",
):
    session.set_current(verse)
    bible.set_manual_index(verse["ref"])
    verse_payload = _annotate_for_session(verse, session)
    add_to_feed(verse_payload, feed_type)
    await safe_send(ws, {
        "type":       payload_type,
        "transcript": query,
        "results":    [verse_payload],
        "feed":       list(scripture_feed),
        "is_exact":   True,
        "confidence": confidence,
        "openai":     openai,
        "transcript_finalized": transcript_finalized,
        "is_interim": is_interim,
        "state":      session.payload(),
    })

# ── Whisper helper ────────────────────────────────────────────
def transcribe_whisper(raw_bytes: bytes) -> Optional[str]:
    if not WHISPER_AVAILABLE or whisper_model is None: return None
    if not raw_bytes or len(raw_bytes) < 3200:
        print("⚠️ Whisper: audio chunk too small, skipped")
        return None
    try:
        import wave
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
                wf.writeframes(raw_bytes)
        result = whisper_model.transcribe(
            tmp_path, language="en", fp16=False,
            initial_prompt="Bible scripture reading. Verses like John 3:16, Psalm 23:1, Genesis 1:1."
        )
        os.unlink(tmp_path)
        return result.get("text", "").strip()
    except Exception as e:
        print(f"Whisper error: {e}")
        return None

# ── OpenAI two-stage ranker ───────────────────────────────────
# Called ONLY on final transcripts — never on interim words.
# Returns: { exact, paraphrase, confidence, reason }
# All values may be None — caller handles it.

_RANK_SYSTEM = """You are a Bible verse detection assistant for a live church presentation system.
You will receive:
  • transcript  — exactly what the preacher said (may be a quote or paraphrase)
  • candidates  — a numbered list of KJV verses retrieved by keyword search

Return ONLY a JSON object with this exact shape (no markdown, no extra text):
{
  "exact":      "<ref>" or null,
  "paraphrase": "<ref>" or null,
  "confidence": <float 0.0–1.0>,
  "reason":     "<one short sentence>"
}

Rules:
- "exact"      = preacher is CITING this verse by reference OR quoting it verbatim.
                 null if they are not quoting directly.
- "paraphrase" = candidate whose MEANING best fits what the preacher said.
                 May equal "exact". null if nothing matches well.
- "confidence" = your certainty. Use < 0.5 for ambiguous or off-topic speech.
- If confidence < 0.45, set both to null and explain in "reason".
- NEVER invent a verse not in the candidate list."""

async def openai_rank_verses(transcript: str, candidates: List[Dict]) -> Dict:
    _null = {"exact": None, "paraphrase": None, "confidence": 0.0, "reason": "OpenAI unavailable"}
    if not OPENAI_AVAILABLE or not _openai_client or not candidates:
        return _null

    numbered = "\n".join(
        f"{i+1}. [{v['ref']}] {v['text']}"
        for i, v in enumerate(candidates[:8])
    )
    user_msg = f'transcript: "{transcript}"\n\ncandidates:\n{numbered}'

    try:
        resp = await _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _RANK_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=120,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw    = resp.choices[0].message.content.strip()
        result = json.loads(raw)
        out = {
            "exact":      result.get("exact")      or None,
            "paraphrase": result.get("paraphrase") or None,
            "confidence": float(result.get("confidence", 0.0)),
            "reason":     result.get("reason", ""),
        }
        print(
            f"🤖 GPT-4o-mini | conf={out['confidence']:.2f} | "
            f"exact={out['exact']} | para={out['paraphrase']} | {out['reason']}"
        )
        return out
    except json.JSONDecodeError as je:
        print(f"⚠  OpenAI bad JSON: {je}")
    except Exception as e:
        print(f"⚠  OpenAI rank error: {e}")
    return _null

# ── Core query processor ──────────────────────────────────────
async def _legacy_process_query_unused(
    ws: WebSocket,
    query: str,
    session: Optional[SessionState] = None,
    is_interim: bool = False,
    transcript_finalized: bool = False,
):
    """
    Pipeline:
      1. Normalise spoken words → searchable text
      2. Exact reference lookup  → exact_match  (→ Detected Verses + Live screen)
      3. Keyword retrieval       → top-K candidates
      4. [Optional] OpenAI rank  → best exact / paraphrase with confidence score
      5. Send result             → fuzzy_match  (→ Paraphrase panel only)

    Interim calls (is_interim=True) skip exact lookup and OpenAI
    to stay fast — they only update the Paraphrase panel live.
    """
    query = query.strip()
    if not query: return

    clean_query = normalise_query(query)

    # ── "next verse" command ──────────────────────────────────
    if NEXT_VERSE_RE.search(query):
        nxt = bible.get_next_verse()
        if nxt:
            add_to_feed(nxt, "next")
            await safe_send(ws, {
                "type": "next_verse", "transcript": query,
                "results": [nxt], "feed": list(scripture_feed)
            })
        return

    # ── Step 1: Exact reference lookup (final transcripts only) ──
    # This populates Detected Verses. Skipped on interim to avoid
    # false positives while the preacher is mid-sentence.
    if not is_interim:
        exact = exact_lookup(clean_query)
        if exact:
            bible.set_manual_index(exact["ref"])
            add_to_feed(exact, "exact")
            await safe_send(ws, {
                "type":       "exact_match",  # → Detected Verses + Live screen
                "transcript": query,
                "results":    [exact],
                "feed":       list(scripture_feed),
                "is_exact":   True,
                "confidence": 1.0,
                "openai":     False,
            })
            return

    # ── Word count gate ───────────────────────────────────────
    word_threshold = 2 if is_interim else MIN_WORDS
    if len(query.split()) < word_threshold:
        if not is_interim:
            await safe_send(ws, {"type": "partial", "transcript": query})
        return

    # ── Step 2: Keyword retrieval (Stage 1) ───────────────────
    candidates = await asyncio.to_thread(bible.keyword_search, clean_query, TOP_K)

    if not candidates:
        if not is_interim:
            await safe_send(ws, {"type": "no_match", "transcript": query})
        return

    # Interim path: send keyword results immediately, no OpenAI
    if is_interim:
        await safe_send(ws, {
            "type":       "interim_suggestions",
            "transcript": query,
            "results":    [_annotate_for_session(v, session) for v in candidates[:3]],
            "openai":     False,
            "state":      session.payload(),
        })
        return

    # ── Step 3: OpenAI ranking (Stage 2 — final only) ────────
    if OPENAI_AVAILABLE:
        ranking    = await openai_rank_verses(query, candidates)
        confidence = ranking["confidence"]

        # High-confidence exact citation → route to Detected Verses + Live screen
        if ranking["exact"] and confidence >= 0.75:
            verse = _find_verse_by_ref(ranking["exact"], candidates)
            if verse:
                bible.set_manual_index(verse["ref"])
                add_to_feed(verse, "exact")
                await safe_send(ws, {
                    "type":       "exact_match",  # → Detected Verses + Live screen
                    "transcript": query,
                    "results":    [verse],
                    "feed":       list(scripture_feed),
                    "is_exact":   True,
                    "confidence": confidence,
                    "openai":     True,
                })
                return

        # Paraphrase match → route to Paraphrase panel only
        if ranking["paraphrase"] and confidence >= 0.45:
            best = _find_verse_by_ref(ranking["paraphrase"], candidates)
            if best:
                ordered = [best] + [r for r in candidates if r["ref"] != best["ref"]]
                await safe_send(ws, {
                    "type":       "fuzzy_match",  # → Paraphrase panel only
                    "transcript": query,
                    "results":    ordered[:3],
                    "is_exact":   False,
                    "confidence": confidence,
                    "reason":     ranking["reason"],
                    "openai":     True,
                })
                return

        # Low confidence — nothing reliable
        await safe_send(ws, {
            "type":       "no_match",
            "transcript": query,
            "reason":     ranking["reason"],
            "confidence": confidence,
            "openai":     True,
        })

    else:
        # No OpenAI key — send keyword results directly to Paraphrase panel
        await safe_send(ws, {
            "type":       "fuzzy_match",  # → Paraphrase panel only
            "transcript": query,
            "results":    candidates[:3],
            "is_exact":   False,
            "openai":     False,
        })

# ─────────────────────────────────────────────────────────────
# WebSocket — main entry
# ─────────────────────────────────────────────────────────────
async def process_query(
    ws: WebSocket,
    query: str,
    session: Optional[SessionState] = None,
    is_interim: bool = False,
    transcript_finalized: bool = False,
):
    """
    Updated live-query pipeline:
      1. Parse voice commands before search
      2. Normalise spoken words to searchable text
      3. Let exact refs and strong verbatim hits go live immediately
      4. Keep paraphrase/fuzzy matches on the slower confirmation path
    """
    session = session or SessionState()
    query = query.strip()
    if not query:
        return

    intent = parse_voice_intent(query, session)
    if intent:
        await execute_voice_intent(
            ws,
            query,
            intent,
            session,
            is_interim=is_interim,
            transcript_finalized=transcript_finalized,
        )
        return

    lane, direct_verse, pending_message = resolve_direct_reference_lane(query, session)
    if lane == "exact" and direct_verse:
        exact = direct_verse
        session.set_current(exact)
        if not _is_last_feed_ref(exact["ref"]):
            await _send_exact_match(
                ws,
                query,
                exact,
                session,
                confidence=1.0,
                openai=False,
                transcript_finalized=transcript_finalized,
                is_interim=is_interim,
            )
        return
    if lane == "pending":
        await safe_send(ws, {
            "type": "partial_reference",
            "transcript": query,
            "message": pending_message or "Waiting for the rest of the reference.",
            "state": session.payload(),
            "transcript_finalized": transcript_finalized,
            "is_interim": is_interim,
        })
        return

    clean_query = normalise_query(query)

    exact = exact_lookup(clean_query)
    if exact:
        session.set_current(exact)
        if not _is_last_feed_ref(exact["ref"]):
            await _send_exact_match(
                ws,
                query,
                exact,
                session,
                confidence=1.0,
                openai=False,
                transcript_finalized=transcript_finalized,
                is_interim=is_interim,
            )
        return

    word_threshold = 2 if is_interim else MIN_WORDS
    if len(query.split()) < word_threshold:
        if not is_interim:
            await safe_send(ws, {
                "type": "partial",
                "transcript": query,
                "transcript_finalized": transcript_finalized,
            })
        return

    verbatim_fast = bible.find_verbatim_phrase(query)
    if verbatim_fast:
        session.set_current(verbatim_fast)
        await _send_exact_match(
            ws,
            query,
            verbatim_fast,
            session,
            confidence=0.98,
            openai=False,
            transcript_finalized=transcript_finalized,
            is_interim=is_interim,
            payload_type="verbatim_match",
            feed_type="verbatim",
        )
        return

    # ── Lane 2: Semantic search with topic-aware query expansion ──
    # _expand_semantic_query enriches the embedding query when the preacher
    # speaks a recognised church theme (soul winning, Holy Spirit, etc.)
    # so FAISS surfaces topically-relevant verses even when the exact words
    # don't appear in the verse text.
    semantic_query = _expand_semantic_query(clean_query)
    candidates = await asyncio.to_thread(bible.vector_search, semantic_query, TOP_K)
    if not candidates:
        if not is_interim:
            await safe_send(ws, {
                "type": "no_match",
                "transcript": query,
                "transcript_finalized": transcript_finalized,
            })
        return

    verbatim = _quick_verbatim_match(query, candidates)
    if verbatim:
        session.set_current(verbatim)
        if not _is_last_feed_ref(verbatim["ref"]):
            await _send_exact_match(
                ws,
                query,
                verbatim,
                session,
                confidence=0.96,
                openai=False,
                transcript_finalized=transcript_finalized,
                is_interim=is_interim,
                payload_type="verbatim_match",
                feed_type="verbatim",
            )
        return

    annotated_candidates = [_annotate_for_session(v, session) for v in candidates[:3]]

    if is_interim:
        await safe_send(ws, {
            "type":       "interim_suggestions",
            "transcript": query,
            "results":    annotated_candidates,
            "openai":     False,
            "state":      session.payload(),
        })
        return

    confidence = float(candidates[0].get("score", 0.0))
    if bible.vector_status().get("ready") and confidence < VECTOR_MIN_SCORE:
        await safe_send(ws, {
            "type":       "no_match",
            "transcript": query,
            "reason":     "Vector confidence below threshold.",
            "confidence": confidence,
            "openai":     False,
            "vector":     True,
            "transcript_finalized": transcript_finalized,
            "state":      session.payload(),
        })
        return

    await safe_send(ws, {
        "type":       "fuzzy_match",
        "transcript": query,
        "results":    annotated_candidates,
        "is_exact":   False,
        "openai":     False,
        "vector":     bible.vector_status().get("ready"),
        "confidence": confidence,
        "transcript_finalized": transcript_finalized,
        "state":      session.payload(),
    })

@app.websocket("/ws/live")
async def live_ws(ws: WebSocket):
    await ws.accept()
    print("🔌 Desktop client connected")

    engine          = "auto"
    session_key     = ""
    prefetched_audio: List[bytes] = []
    global global_session
    session_state   = global_session   # share with /ws/control so remote NEXT/PREV tracks the latest verse

    try:
        loop          = asyncio.get_running_loop()
        init_deadline = loop.time() + 5.0

        while True:
            remaining = init_deadline - loop.time()
            if remaining <= 0:
                break

            raw = await asyncio.wait_for(ws.receive(), timeout=remaining)

            if raw.get("type") == "websocket.disconnect":
                print("🔌 Client disconnected before init")
                return

            if "bytes" in raw and raw["bytes"]:
                prefetched_audio.append(raw["bytes"])
                continue

            if "text" not in raw or not raw["text"]:
                continue

            try:
                msg = json.loads(raw["text"])
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "init":
                engine      = msg.get("engine", "auto")
                session_key = msg.get("deepgram_key", "").strip()
                break
    except WebSocketDisconnect:
        print("🔌 Client disconnected before session start")
        return
    except Exception:
        pass

    effective_dg_key = session_key or SERVER_DG_KEY

    # Resolve which engine to use
    if engine == "auto":
        engine = "deepgram" if effective_dg_key else ("whisper" if WHISPER_AVAILABLE else "text")
    elif engine == "deepgram" and not effective_dg_key:
        fallback = "whisper" if WHISPER_AVAILABLE else "text"
        await safe_send(ws, {"type": "warning", "message": f"No Deepgram key — using {fallback}."})
        engine = fallback
    elif engine == "whisper" and not WHISPER_AVAILABLE:
        await safe_send(ws, {"type": "warning", "message": "Whisper not installed — text mode."})
        engine = "text"

    key_source = "client" if (session_key and not SERVER_DG_KEY) else ("server" if SERVER_DG_KEY else "none")

    await safe_send(ws, {
        "type": "connected", "message": "In The Beginning is live!",
        "engine": engine, "deepgram": engine == "deepgram",
        "whisper": engine == "whisper", "key_source": key_source,
    })
    print(f"🔌 Engine: {engine} | key_source: {key_source}")

    # Load FAISS lazily — only when transcription actually starts
    await _ensure_vector_index()

    if engine == "deepgram":
        await _run_deepgram(ws, effective_dg_key, session_state, prefetched_audio)
    elif engine == "whisper":
        await _run_whisper(ws, session_state, prefetched_audio)
    else:
        await _run_text(ws, session_state)

# ── Deepgram session ──────────────────────────────────────────
async def _run_deepgram(
    ws: WebSocket,
    dg_key: str,
    session: SessionState,
    prefetched_audio: Optional[List[bytes]] = None,
):
    # Deepgram nova-3 — tuned for fast preachers:
    # • endpointing=250 ms  → detect short silence between bursts quickly
    # • utterance_end_ms=2400 → give a fast preacher 2.4 s of silence before
    #   closing an utterance (was 1600 — too tight for rapid delivery)
    # • filler_words=false → strip "uh/um" so they don't pollute verse lookup
    # • diarize=false → single-speaker mode is faster
    DG_URL = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-3&language=en-US&encoding=linear16"
        "&sample_rate=16000&channels=1"
        "&interim_results=true&punctuate=true&filler_words=false&diarize=false"
        f"&smart_format=true&no_delay=true&endpointing={DEEPGRAM_ENDPOINT_MS}"
        f"&utterance_end_ms={DEEPGRAM_UTTERANCE_END_MS}&vad_events=true"
    )

    import websockets as _ws_lib
    import inspect

    AUDIO_QUEUE_MAX        = 96   # ~6 s of browser audio at 64 ms/chunk
    FINAL_QUERY_QUEUE_MAX  = 16
    INTERIM_QUERY_MIN_SEC  = 0.30
    INTERIM_QUERY_MIN_WORDS = 3

    audio_q  = asyncio.Queue(maxsize=AUDIO_QUEUE_MAX)
    final_query_q = asyncio.Queue(maxsize=FINAL_QUERY_QUEUE_MAX)
    stop_evt = asyncio.Event()
    prefetched_audio = prefetched_audio or []

    final_segments: List[str] = []
    final_segment_keys = set()
    interim_query_task: Optional[asyncio.Task] = None
    last_interim_query = ""
    last_interim_query_at = 0.0
    last_live_transcript = ""
    last_drop_notice_at = 0.0

    def _segment_key(msg: Dict, tx: str):
        return (
            round(float(msg.get("start", 0.0)), 3),
            round(float(msg.get("duration", 0.0)), 3),
            tx,
        )

    def _append_final_segment(msg: Dict, tx: str):
        clean_tx = tx.strip()
        if not clean_tx:
            return
        key = _segment_key(msg, clean_tx)
        if key in final_segment_keys:
            return
        final_segment_keys.add(key)
        final_segments.append(clean_tx)

    def _compose_utterance(current_tx: str = "") -> str:
        parts = [seg.strip() for seg in final_segments if seg.strip()]
        clean_current = current_tx.strip()
        if clean_current and (not parts or parts[-1] != clean_current):
            parts.append(clean_current)
        return " ".join(parts).strip()

    def _consume_utterance(fallback_tx: str = "") -> str:
        utterance = _compose_utterance(fallback_tx)
        final_segments.clear()
        final_segment_keys.clear()
        return utterance

    def _finish_task(task: asyncio.Task):
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"DG interim task error: {e}")

    def _cancel_interim_query():
        nonlocal interim_query_task
        if interim_query_task and not interim_query_task.done():
            interim_query_task.cancel()
        interim_query_task = None

    def _should_schedule_interim_query(text: str) -> bool:
        if parse_voice_intent(text, session):
            return True
        if re.search(r"\d+\s*:?\s*\d+", normalise_query(text)):
            return True
        return len(text.split()) >= INTERIM_QUERY_MIN_WORDS

    def _schedule_interim_query(text: str):
        nonlocal interim_query_task
        _cancel_interim_query()
        interim_query_task = asyncio.create_task(process_query(ws, text, session, is_interim=True))
        interim_query_task.add_done_callback(_finish_task)

    async def _enqueue_audio(chunk: bytes):
        nonlocal last_drop_notice_at
        if audio_q.full():
            dropped = 0
            while audio_q.full():
                try:
                    audio_q.get_nowait()
                    dropped += 1
                except asyncio.QueueEmpty:
                    break
            now = asyncio.get_event_loop().time()
            if dropped and (now - last_drop_notice_at) >= 2.0:
                print(f"⚠️ Deepgram backlog detected — dropped {dropped} stale audio chunk(s) to stay live")
                last_drop_notice_at = now
        await audio_q.put(chunk)

    async def _submit_final_query(text: str):
        clean_text = text.strip()
        if not clean_text:
            return

        await safe_send(ws, {
            "type": "final_transcript",
            "transcript": clean_text,
            "transcript_finalized": True,
        })

        if final_query_q.full():
            try:
                final_query_q.get_nowait()
                final_query_q.task_done()
            except asyncio.QueueEmpty:
                pass
            print("Deepgram final-query backlog detected - dropped oldest verse lookup to stay live")

        try:
            final_query_q.put_nowait(clean_text)
        except asyncio.QueueFull:
            print("Deepgram final-query queue still full - skipped verse lookup")

    async def final_query_worker():
        while not stop_evt.is_set():
            query = await final_query_q.get()
            try:
                if query is None:
                    return
                await process_query(
                    ws,
                    query,
                    session,
                    is_interim=False,
                    transcript_finalized=True,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"DG final query error: {e}")
            finally:
                final_query_q.task_done()

    # ── Keepalive ─────────────────────────────────────────────
    # Deepgram closes with net0001 after 10 s of silence.
    # We send a KeepAlive JSON message every 8 s when no audio is flowing.
    KEEPALIVE     = json.dumps({"type": "KeepAlive"})
    KEEPALIVE_SEC = 5

    async def fwd(dg_ws):
        last_sent = asyncio.get_event_loop().time()
        try:
            while not stop_evt.is_set():
                try:
                    chunk = await asyncio.wait_for(audio_q.get(), timeout=1.0)
                    if chunk is None: break
                    await dg_ws.send(chunk)
                    last_sent = asyncio.get_event_loop().time()
                except asyncio.TimeoutError:
                    # Queue empty — send KeepAlive if due
                    if asyncio.get_event_loop().time() - last_sent >= KEEPALIVE_SEC:
                        try:
                            await dg_ws.send(KEEPALIVE)
                            last_sent = asyncio.get_event_loop().time()
                            print("💓 Deepgram KeepAlive sent")
                        except Exception as e:
                            print(f"KeepAlive failed: {e}")
                            break
        except Exception as e:
            print(f"DG fwd error: {e}")

    async def rcv(dg_ws):
        nonlocal last_interim_query, last_interim_query_at, last_live_transcript
        try:
            async for raw in dg_ws:
                if stop_evt.is_set(): break
                try:
                    msg = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
                    msg_type = msg.get("type", "Results")

                    if msg_type == "UtteranceEnd":
                        _cancel_interim_query()
                        utterance = _consume_utterance()
                        if utterance:
                            last_interim_query = ""
                            last_live_transcript = ""
                            await _submit_final_query(utterance)
                        continue

                    if msg_type != "Results":
                        continue

                    alts       = msg.get("channel", {}).get("alternatives", [{}])
                    tx         = alts[0].get("transcript", "").strip() if alts else ""
                    is_final   = msg.get("is_final", False)
                    speech_fin = msg.get("speech_final", False)

                    if is_final and tx:
                        _append_final_segment(msg, tx)

                    live_tx = _compose_utterance("" if is_final else tx)
                    if live_tx and live_tx != last_live_transcript:
                        # Keep the transcript live until Deepgram confirms the
                        # speaker has actually paused.
                        await safe_send(ws, {
                            "type":       "interim",
                            "transcript": live_tx,
                            "is_final":   False,
                        })
                        last_live_transcript = live_tx

                    if speech_fin:
                        _cancel_interim_query()
                        utterance = _consume_utterance(tx if not is_final else "")
                        if utterance:
                            last_interim_query = ""
                            last_live_transcript = ""
                            await _submit_final_query(utterance)
                    elif live_tx:
                        now = asyncio.get_event_loop().time()
                        if (
                            live_tx != last_interim_query
                            and _should_schedule_interim_query(live_tx)
                            and (now - last_interim_query_at) >= INTERIM_QUERY_MIN_SEC
                        ):
                            last_interim_query = live_tx
                            last_interim_query_at = now
                            _schedule_interim_query(live_tx)

                except Exception as e:
                    print(f"DG message handling error: {e}")
                    continue
        except Exception as e:
            if not stop_evt.is_set():
                print(f"DG rcv error: {e}")
                await safe_send(ws, {"type": "dg_error", "error": str(e)})

    # ── Connect to Deepgram ───────────────────────────────────
    try:
        _connect_sig = inspect.signature(_ws_lib.connect)
        _header_kwarg = (
            "additional_headers" if "additional_headers" in _connect_sig.parameters
            else "extra_headers"
        )

        async with _ws_lib.connect(
            DG_URL,
            **{_header_kwarg: {"Authorization": f"Token {dg_key}"}},
            ping_interval=20, ping_timeout=20,
        ) as dg_ws:
            await safe_send(ws, {"type": "dg_ready", "message": "🎙️ Deepgram connected — speak now"})
            print("✅ Deepgram stream open")
            fwd_t = asyncio.create_task(fwd(dg_ws))
            rcv_t = asyncio.create_task(rcv(dg_ws))
            final_worker_t = asyncio.create_task(final_query_worker())
            try:
                for chunk in prefetched_audio:
                    await _enqueue_audio(chunk)

                while True:
                    data = await ws.receive()
                    if "bytes" in data and data["bytes"]:
                        await _enqueue_audio(data["bytes"])
                    elif "text" in data:
                        try:
                            msg = json.loads(data["text"])
                            t   = msg.get("type")
                            if t == "stop":
                                break
                            elif t == "next":
                                await execute_voice_intent(
                                    ws,
                                    msg.get("text", "next verse"),
                                    {"kind": "navigation", "action": "next_verse"},
                                    session,
                                    transcript_finalized=True,
                                    dedupe=False,
                                )
                            elif t == "previous":
                                await execute_voice_intent(
                                    ws,
                                    msg.get("text", "previous verse"),
                                    {"kind": "navigation", "action": "previous_verse"},
                                    session,
                                    transcript_finalized=True,
                                    dedupe=False,
                                )
                            elif t == "set_translation":
                                code = _translation_from_phrase(msg.get("translation", ""))
                                if code:
                                    await execute_voice_intent(
                                        ws,
                                        f"switch to {code}",
                                        {"kind": "translation", "action": "set_translation", "translation": code},
                                        session,
                                        transcript_finalized=True,
                                        dedupe=False,
                                    )
                            elif t == "get_feed":
                                await safe_send(ws, {"type": "feed", "feed": list(scripture_feed)})
                            elif t == "set_index":
                                ref = msg.get("ref", "")
                                bible.set_manual_index(ref)
                                verse = bible.find_by_ref(ref)
                                if verse:
                                    session.set_current(verse)
                            elif t == "transcript":
                                await process_query(ws, msg.get("text", ""), session)
                        except json.JSONDecodeError:
                            pass
            except WebSocketDisconnect:
                print("🔌 Browser disconnected during Deepgram session")
            finally:
                stop_evt.set()
                _cancel_interim_query()
                try:
                    while audio_q.full():
                        audio_q.get_nowait()
                    audio_q.put_nowait(None)
                except asyncio.QueueFull:
                    pass
                fwd_t.cancel()
                rcv_t.cancel()
                final_worker_t.cancel()
                try: await dg_ws.send(json.dumps({"type": "CloseStream"}))
                except Exception: pass
                await asyncio.gather(fwd_t, rcv_t, final_worker_t, return_exceptions=True)
                print("🔌 Deepgram closed")

    except Exception as e:
        print(f"❌ Deepgram connection failed: {e}")
        # Only try to notify if the browser socket is still open
        await safe_send(ws, {
            "type":    "dg_error",
            "error":   str(e),
            "message": "Deepgram failed — check your API key or switch to Offline Mode.",
        })
        # Do NOT fall back to Whisper automatically — browser socket may be closed
        # The user can manually switch to Offline mode in the UI.

# ── Whisper session ───────────────────────────────────────────
async def _run_whisper(ws: WebSocket, session: SessionState, prefetched_audio: Optional[List[bytes]] = None):
    # Load the Whisper model lazily — only on first Whisper session.
    # This keeps boot memory light; the model (~150MB) stays resident after first load.
    if WHISPER_AVAILABLE and whisper_model is None:
        await safe_send(ws, {"type": "status", "message": "⏳ Loading Whisper model (first use)…"})
        await _ensure_whisper_model()

    # Guard: check socket is still alive before sending anything
    if not await safe_send(ws, {"type": "whisper_ready", "message": "🎙️ Whisper ready — speak now (Offline Mode)"}):
        print("⚠  Whisper: socket already closed, aborting")
        return
    print("🎙️ Whisper session started")

    audio_buffer    = bytearray()
    CHUNK_THRESHOLD = 16000 * 2 * 1  # ~1 second — transcribe as soon as audio arrives
    loop            = asyncio.get_event_loop()
    prefetched_audio = prefetched_audio or []

    async def flush():
        nonlocal audio_buffer
        if len(audio_buffer) < 3200:
            audio_buffer = bytearray()
            return
        chunk = bytes(audio_buffer)
        audio_buffer = bytearray()
        tx = await loop.run_in_executor(None, transcribe_whisper, chunk)
        if tx:
            print(f"🎙️ Whisper: {tx}")
            ok = await safe_send(ws, {"type": "interim", "transcript": tx})
            if ok:
                await process_query(ws, tx, session)

    for chunk in prefetched_audio:
        audio_buffer.extend(chunk)
        if len(audio_buffer) >= CHUNK_THRESHOLD:
            await flush()

    try:
        while True:
            data = await ws.receive()
            if "bytes" in data and data["bytes"]:
                audio_buffer.extend(data["bytes"])
                if len(audio_buffer) >= CHUNK_THRESHOLD:
                    await flush()
            elif "text" in data:
                try:
                    msg = json.loads(data["text"])
                    t   = msg.get("type")
                    if t == "stop":
                        if audio_buffer: await flush()
                        break
                    elif t == "flush":
                        await flush()
                    elif t == "next":
                        await execute_voice_intent(
                            ws,
                            msg.get("text", "next verse"),
                            {"kind": "navigation", "action": "next_verse"},
                            session,
                            transcript_finalized=True,
                            dedupe=False,
                        )
                    elif t == "previous":
                        await execute_voice_intent(
                            ws,
                            msg.get("text", "previous verse"),
                            {"kind": "navigation", "action": "previous_verse"},
                            session,
                            transcript_finalized=True,
                            dedupe=False,
                        )
                    elif t == "set_translation":
                        code = _translation_from_phrase(msg.get("translation", ""))
                        if code:
                            await execute_voice_intent(
                                ws,
                                f"switch to {code}",
                                {"kind": "translation", "action": "set_translation", "translation": code},
                                session,
                                transcript_finalized=True,
                                dedupe=False,
                            )
                    elif t == "get_feed":
                        await safe_send(ws, {"type": "feed", "feed": list(scripture_feed)})
                    elif t == "set_index":
                        ref = msg.get("ref", "")
                        bible.set_manual_index(ref)
                        verse = bible.find_by_ref(ref)
                        if verse:
                            session.set_current(verse)
                    elif t == "transcript":
                        await process_query(ws, msg.get("text", ""), session)
                except json.JSONDecodeError:
                    pass
    except WebSocketDisconnect:
        print("🔌 Browser disconnected during Whisper session")
    finally:
        print("🔌 Whisper session closed")

# ── Text-only fallback ────────────────────────────────────────
async def _run_text(ws: WebSocket, session: SessionState):
    print("⌨️  Text-only session")
    try:
        while True:
            data = await ws.receive()
            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                    t   = msg.get("type")
                    if t == "transcript":
                        await process_query(ws, msg.get("text", ""), session)
                    elif t == "next":
                        await execute_voice_intent(
                            ws,
                            msg.get("text", "next verse"),
                            {"kind": "navigation", "action": "next_verse"},
                            session,
                            transcript_finalized=True,
                            dedupe=False,
                        )
                    elif t == "previous":
                        await execute_voice_intent(
                            ws,
                            msg.get("text", "previous verse"),
                            {"kind": "navigation", "action": "previous_verse"},
                            session,
                            transcript_finalized=True,
                            dedupe=False,
                        )
                    elif t == "set_translation":
                        code = _translation_from_phrase(msg.get("translation", ""))
                        if code:
                            await execute_voice_intent(
                                ws,
                                f"switch to {code}",
                                {"kind": "translation", "action": "set_translation", "translation": code},
                                session,
                                transcript_finalized=True,
                                dedupe=False,
                            )
                    elif t == "get_feed":
                        await safe_send(ws, {"type": "feed", "feed": list(scripture_feed)})
                    elif t == "set_index":
                        ref = msg.get("ref", "")
                        bible.set_manual_index(ref)
                        verse = bible.find_by_ref(ref)
                        if verse:
                            session.set_current(verse)
                    elif t == "stop":
                        break
                except json.JSONDecodeError:
                    pass
    except WebSocketDisconnect:
        print("🔌 Browser disconnected during text session")
    finally:
        print("🔌 Text-only session closed")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main_fixed:app", host="0.0.0.0", port=port, reload=False)