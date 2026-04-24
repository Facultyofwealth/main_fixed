# ============================================================
#  In The Beginning — FastAPI Backend  (Deepgram-first edition)
#  Run:  uvicorn main_fixed:app --reload --port 8000
# ============================================================

import os

# ── Load .env BEFORE anything else ───────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_path   = os.path.join(_script_dir, ".env")
if not os.path.exists(_env_path):
    _env_path = os.path.join(os.getcwd(), ".env")

from dotenv import load_dotenv
load_dotenv(dotenv_path=_env_path, override=True)
print(f"📄 .env loaded from: {_env_path}  (exists={os.path.exists(_env_path)})")

import time, json, asyncio, re, tempfile
from collections import deque
from typing import List, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.websockets import WebSocketState

# ── Deepgram key ──────────────────────────────────────────────
SERVER_DG_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()
if SERVER_DG_KEY and SERVER_DG_KEY != "your_deepgram_api_key_here":
    print(f"🔑 Server DEEPGRAM_API_KEY found: {SERVER_DG_KEY[:8]}…")
else:
    SERVER_DG_KEY = ""
    print("ℹ  No server Deepgram key — clients may supply their own.")

# ── OpenAI (optional — two-stage ranking) ────────────────────
#  Stage 1: keyword search pulls top-K candidates from KJV JSON  (always runs)
#  Stage 2: GPT-4o-mini re-ranks, picks best exact/paraphrase,   (only with key)
#           returns a confidence score, rejects weak matches.
# ─────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_AVAILABLE = False
_openai_client   = None

if OPENAI_API_KEY and OPENAI_API_KEY not in ("", "your_openai_api_key_here"):
    try:
        from openai import AsyncOpenAI
        _openai_client   = AsyncOpenAI(api_key=OPENAI_API_KEY)
        OPENAI_AVAILABLE = True
        print(f"✅ OpenAI ready — two-stage ranking active (key: {OPENAI_API_KEY[:8]}…)")
    except ImportError:
        print("⚠  openai package not installed — run: pip install openai")
    except Exception as _oe:
        print(f"⚠  OpenAI init failed: {_oe}")
else:
    print("ℹ  No OPENAI_API_KEY — keyword-only ranking (Stage 1 only).")

# ── Whisper (offline fallback) ────────────────────────────────
WHISPER_AVAILABLE = False
whisper_model     = None

try:
    import whisper
    whisper_model     = whisper.load_model("tiny")
    WHISPER_AVAILABLE = True
    print("✅ Whisper 'tiny' model loaded")
except ImportError:
    print("⚠  Whisper not installed — run: pip install openai-whisper")
except Exception as e:
    print(f"⚠  Whisper failed to load: {e}")

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
TOP_K     = 5   # retrieve top-5 for Stage 2; display top-3
MIN_WORDS = 3

NEXT_VERSE_RE = re.compile(
    r"\b(next\s*verse|next\s*scripture|move\s*on|go\s*to\s*next|proceed|the\s*next\s*one)\b",
    re.IGNORECASE,
)

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

class BibleIndex:
    def __init__(self):
        self.verses: List[Dict] = []
        self._last_idx: Optional[int] = None

    def load(self):
        candidates = [
            "kjv.json/kjv-master/json/verses-1769.json",
            "kjv.json", "verses-1769.json", "bible.json",
        ]
        for path in candidates:
            verses = self._try_load(path)
            if verses:
                self.verses = verses
                print(f"✅ Loaded {len(self.verses)} verses from {path}")
                return
        self.verses = self._sample()
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
        q_words = set(re.sub(r"[^\w\s]", "", query.lower()).split()) - STOP_WORDS
        if not q_words: return []
        scored = []
        for i, v in enumerate(self.verses):
            searchable = v["ref"].lower() + " " + v["text"].lower()
            v_words    = set(re.sub(r"[^\w\s]", "", searchable).split())
            overlap    = len(q_words & v_words)
            if overlap:
                score = round(overlap / max(len(q_words), 1), 3)
                scored.append({**v, "score": score, "_idx": i})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def get_next_verse(self) -> Optional[Dict]:
        if self._last_idx is None: return None
        nxt = self._last_idx + 1
        if nxt < len(self.verses):
            self._last_idx = nxt
            v = self.verses[nxt]
            return {**v, "score": 1.0, "_idx": nxt}
        return None

    def set_manual_index(self, ref: str):
        ref_lower = ref.lower().strip()
        for i, v in enumerate(self.verses):
            if v["ref"].lower().strip() == ref_lower:
                self._last_idx = i
                print(f"📌 Index set to [{i}] {v['ref']}")
                return

bible = BibleIndex()

@app.on_event("startup")
async def startup():
    bible.load()
    dg  = f"✅ ({SERVER_DG_KEY[:8]}…)" if SERVER_DG_KEY else "⚠  no key"
    ai  = "✅ GPT-4o-mini" if OPENAI_AVAILABLE else "❌ disabled"
    wh  = "✅" if WHISPER_AVAILABLE else "❌"
    print(f"🚀 Ready | Deepgram: {dg} | Whisper: {wh} | OpenAI ranking: {ai}")
    print(f"📖 Frontend: {FRONTEND_FILE or 'NOT FOUND'}")

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

# ── Health ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":      "ok",
        "verse_count": len(bible.verses),
        "deepgram":    bool(SERVER_DG_KEY),
        "whisper":     WHISPER_AVAILABLE,
        "openai":      OPENAI_AVAILABLE,
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
    results = bible.keyword_search(req.text)
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

@app.get("/chapter")
def get_chapter(book: str, chapter: int):
    prefix = f"{book} {chapter}:".lower()
    verses = [v for v in bible.verses if v["ref"].lower().startswith(prefix)]
    return {"book": book, "chapter": chapter, "verses": verses}

class LiveSearchReq(BaseModel):
    text: str

@app.post("/live-search")
async def live_search(req: LiveSearchReq):
    text = req.text.strip()
    if not text or len(text) < 3:
        return {"results": [], "is_exact": False, "query": text}
    clean = normalise_query(text)
    exact = exact_lookup(clean)
    if exact:
        return {"results": [exact], "is_exact": True, "query": text}
    results = bible.keyword_search(clean, k=TOP_K)
    return {"results": results, "is_exact": False, "query": text}

# ── Helpers ───────────────────────────────────────────────────
def exact_lookup(ref: str) -> Optional[Dict]:
    """Match a reference string against the Bible index — two passes."""
    ref_lower = ref.lower().strip()
    # Pass 1: exact string match
    for v in bible.verses:
        if v["ref"].lower().strip() == ref_lower:
            return {**v, "score": 1.0}
    # Pass 2: ignore spaces (handles "John3:16" vs "John 3:16")
    ref_nospace = ref_lower.replace(" ", "")
    for v in bible.verses:
        if v["ref"].lower().replace(" ", "") == ref_nospace:
            return {**v, "score": 1.0}
    return None

def normalise_query(query: str) -> str:
    """
    Translate spoken words → searchable numbers/symbols.
    "First Corinthians ten verse twelve" → "1 corinthians 10:12"
    """
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

def _find_verse_by_ref(ref: Optional[str], candidates: List[Dict]) -> Optional[Dict]:
    if not ref: return None
    ref_l = ref.lower().strip()
    for v in candidates:
        if v["ref"].lower().strip() == ref_l:
            return v
    return None

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
async def process_query(ws: WebSocket, query: str, is_interim: bool = False):
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
    candidates = bible.keyword_search(clean_query, k=TOP_K)

    if not candidates:
        if not is_interim:
            await safe_send(ws, {"type": "no_match", "transcript": query})
        return

    # Interim path: send keyword results immediately, no OpenAI
    if is_interim:
        await safe_send(ws, {
            "type":       "interim_suggestions",
            "transcript": query,
            "results":    candidates[:3],
            "openai":     False,
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
@app.websocket("/ws/live")
async def live_ws(ws: WebSocket):
    await ws.accept()
    print("🔌 Client connected")

    engine      = "auto"
    session_key = ""

    try:
        raw = await asyncio.wait_for(ws.receive(), timeout=5.0)
        if "text" in raw:
            msg = json.loads(raw["text"])
            if msg.get("type") == "init":
                engine      = msg.get("engine", "auto")
                session_key = msg.get("deepgram_key", "").strip()
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

    if engine == "deepgram":
        await _run_deepgram(ws, effective_dg_key)
    elif engine == "whisper":
        await _run_whisper(ws)
    else:
        await _run_text(ws)

# ── Deepgram session ──────────────────────────────────────────
async def _run_deepgram(ws: WebSocket, dg_key: str):
    DG_URL = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-2&language=en-US&encoding=linear16"
        "&sample_rate=16000&channels=1"
        "&interim_results=true&punctuate=true"
        "&smart_format=true&endpointing=500"
    )

    import websockets as _ws_lib
    import inspect

    audio_q  = asyncio.Queue()
    stop_evt = asyncio.Event()

    # ── Keepalive ─────────────────────────────────────────────
    # Deepgram closes with net0001 after 10 s of silence.
    # We send a KeepAlive JSON message every 8 s when no audio is flowing.
    KEEPALIVE     = json.dumps({"type": "KeepAlive"}).encode()
    KEEPALIVE_SEC = 8

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
        try:
            async for raw in dg_ws:
                if stop_evt.is_set(): break
                try:
                    msg        = json.loads(raw)
                    alts       = msg.get("channel", {}).get("alternatives", [{}])
                    tx         = alts[0].get("transcript", "").strip() if alts else ""
                    is_final   = msg.get("is_final", False)
                    speech_fin = msg.get("speech_final", False)
                    if not tx: continue

                    # Send interim words to screen immediately
                    await safe_send(ws, {
                        "type":       "interim",
                        "transcript": tx,
                        "is_final":   is_final,
                    })

                    # Interim: update Paraphrase panel live (keyword only, no OpenAI)
                    if not is_final and not speech_fin:
                        await process_query(ws, tx, is_interim=True)

                    # Final: run full pipeline (exact lookup + OpenAI ranking)
                    if speech_fin or (is_final and len(tx.split()) >= MIN_WORDS):
                        await process_query(ws, tx, is_interim=False)

                except Exception:
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
            try:
                while True:
                    data = await ws.receive()
                    if "bytes" in data and data["bytes"]:
                        await audio_q.put(data["bytes"])
                    elif "text" in data:
                        try:
                            msg = json.loads(data["text"])
                            t   = msg.get("type")
                            if t == "stop":
                                break
                            elif t == "next":
                                nxt = bible.get_next_verse()
                                if nxt:
                                    add_to_feed(nxt, "next")
                                    await safe_send(ws, {"type": "next_verse", "results": [nxt], "feed": list(scripture_feed)})
                            elif t == "get_feed":
                                await safe_send(ws, {"type": "feed", "feed": list(scripture_feed)})
                            elif t == "set_index":
                                bible.set_manual_index(msg.get("ref", ""))
                            elif t == "transcript":
                                await process_query(ws, msg.get("text", ""))
                        except json.JSONDecodeError:
                            pass
            except WebSocketDisconnect:
                print("🔌 Browser disconnected during Deepgram session")
            finally:
                stop_evt.set()
                await audio_q.put(None)
                fwd_t.cancel()
                rcv_t.cancel()
                try: await dg_ws.send(json.dumps({"type": "CloseStream"}))
                except Exception: pass
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
async def _run_whisper(ws: WebSocket):
    # Guard: check socket is still alive before sending anything
    if not await safe_send(ws, {"type": "whisper_ready", "message": "🎙️ Whisper ready — speak now (Offline Mode)"}):
        print("⚠  Whisper: socket already closed, aborting")
        return
    print("🎙️ Whisper session started")

    audio_buffer  = bytearray()
    CHUNK_THRESHOLD = 16000 * 2 * 3  # ~3 seconds of audio before transcribing
    loop = asyncio.get_event_loop()

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
                await process_query(ws, tx)

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
                        nxt = bible.get_next_verse()
                        if nxt:
                            add_to_feed(nxt, "next")
                            await safe_send(ws, {"type": "next_verse", "results": [nxt], "feed": list(scripture_feed)})
                    elif t == "get_feed":
                        await safe_send(ws, {"type": "feed", "feed": list(scripture_feed)})
                    elif t == "set_index":
                        bible.set_manual_index(msg.get("ref", ""))
                    elif t == "transcript":
                        await process_query(ws, msg.get("text", ""))
                except json.JSONDecodeError:
                    pass
    except WebSocketDisconnect:
        print("🔌 Browser disconnected during Whisper session")
    finally:
        print("🔌 Whisper session closed")

# ── Text-only fallback ────────────────────────────────────────
async def _run_text(ws: WebSocket):
    print("⌨️  Text-only session")
    try:
        while True:
            data = await ws.receive()
            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                    t   = msg.get("type")
                    if t == "transcript":
                        await process_query(ws, msg.get("text", ""))
                    elif t == "next":
                        nxt = bible.get_next_verse()
                        if nxt:
                            add_to_feed(nxt, "next")
                            await safe_send(ws, {"type": "next_verse", "results": [nxt], "feed": list(scripture_feed)})
                    elif t == "get_feed":
                        await safe_send(ws, {"type": "feed", "feed": list(scripture_feed)})
                    elif t == "set_index":
                        bible.set_manual_index(msg.get("ref", ""))
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
    uvicorn.run("main_fixed:app", host="0.0.0.0", port=8000, reload=True)