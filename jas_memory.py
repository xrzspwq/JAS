#!/usr/bin/env python3
"""
jas_memory.py - Jas's long-term memory.

Short notes ("the user prefers dark mode", "project X lives in D:/code/x") are stored in a small SQLite
file and retrieved BY MEANING with your embedding model (the "embed" model in llama-swap).
If the embedding model is unavailable, it falls back to keyword matching, so memory never blocks Jas.

    ~/.jas/memory.db        (override with the JAS_MEMORY environment variable)
    JAS_EMBED_MODEL         llama-swap model name used for embeddings (default: embed)

Used by jas.py; you normally don't run this file. `python jas_memory.py` prints your notes.
"""
import array
import os
import re
import sqlite3
import time
from datetime import datetime
from operator import mul
from pathlib import Path

DB_PATH = Path(os.environ.get("JAS_MEMORY") or (Path.home() / ".jas" / "memory.db"))
EMBED_MODEL = os.environ.get("JAS_EMBED_MODEL", "embed")
# Qwen3-Embedding works best when the QUERY (not the stored notes) carries an instruction.
QUERY_PREFIX = ("Instruct: Given a message from the user, retrieve stored notes about the user, their PC, "
                "projects, preferences and past decisions that are relevant to it\nQuery: ")
DUP_THRESHOLD = 0.93          # cosine similarity above which a new note counts as a duplicate
MAX_NOTE_CHARS = 500
SECRET_RX = re.compile(r"(?i)\b(pass(word|wd)?|api[_ -]?key|secret|token|private[_ -]?key)\b\s*(is|[:=])\s*\S+|\bsk-[A-Za-z0-9_-]{16,}")
STOP = set("the and for are but not you your with that this have has had was were will would can could should "
           "from they them then than into about what when where which who how why all any some out use using "
           "its our their his her she him get got let just also very more most".split())


def _tokens(s: str) -> set:
    return {t for t in re.findall(r"[a-z0-9_]{3,}", s.lower())} - STOP


def _keyword_score(qtok: set, text: str) -> float:
    shared = len(qtok & _tokens(text))
    if shared >= 2 or (shared >= 1 and len(qtok) <= 2):
        return shared / max(len(qtok), 1)
    return 0.0


class Memory:
    def __init__(self, client, path: Path = None, embed_model: str = None):
        self.client = client
        self.embed_model = embed_model or EMBED_MODEL
        self.path = Path(path or DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.execute("""CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, kind TEXT NOT NULL,
            text TEXT NOT NULL, dim INTEGER, vec BLOB)""")
        self.db.commit()
        self.last_error = ""
        self._fails = 0
        self._skip_until = 0.0

    # ---------------------------------------------------------------- embeddings
    def _embed(self, text: str, query: bool = False):
        """Unit-length float vector, or None if the embedding model is unavailable."""
        if time.time() < self._skip_until:
            return None
        try:
            payload = (QUERY_PREFIX + text) if query else text
            r = self.client.embeddings.create(model=self.embed_model, input=payload[:2000])
            vec = list(r.data[0].embedding)
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            self._fails = 0
            return array.array("f", [x / norm for x in vec])
        except Exception as e:                   # server down, model missing, ...
            self.last_error = f"{type(e).__name__}: {e}"
            self._fails += 1
            if self._fails >= 2:                 # stop hammering a dead endpoint for a minute
                self._skip_until = time.time() + 60
            return None

    @staticmethod
    def _load(blob) -> array.array:
        v = array.array("f")
        v.frombytes(blob)
        return v

    # ---------------------------------------------------------------- write
    def add(self, text: str, kind: str = "note"):
        """-> (note_id or None, status) with status 'saved', 'duplicate' or a reason it was refused."""
        text = " ".join((text or "").split())[:MAX_NOTE_CHARS]
        if len(text) < 8:
            return None, "too short to be useful"
        if SECRET_RX.search(text):
            return None, "looks like a password/API key, so it was NOT saved"
        row = self.db.execute("SELECT id FROM notes WHERE lower(text)=?", (text.lower(),)).fetchone()
        if row:
            return row[0], "duplicate"
        vec = self._embed(text)
        if vec is not None:
            for nid, dim, blob in self.db.execute("SELECT id, dim, vec FROM notes WHERE vec IS NOT NULL"):
                if dim == len(vec) and sum(map(mul, vec, self._load(blob))) >= DUP_THRESHOLD:
                    return nid, "duplicate"
        cur = self.db.execute(
            "INSERT INTO notes (ts, kind, text, dim, vec) VALUES (?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), kind, text,
             len(vec) if vec is not None else None, vec.tobytes() if vec is not None else None))
        self.db.commit()
        return cur.lastrowid, "saved"

    def forget(self, note_id: int) -> bool:
        cur = self.db.execute("DELETE FROM notes WHERE id=?", (int(note_id),))
        self.db.commit()
        return cur.rowcount > 0

    def backfill(self, limit: int = 50) -> int:
        """Embed notes that were saved while the embedding model was unavailable."""
        done = 0
        for nid, text in self.db.execute("SELECT id, text FROM notes WHERE vec IS NULL LIMIT ?", (limit,)).fetchall():
            vec = self._embed(text)
            if vec is None:
                break
            self.db.execute("UPDATE notes SET dim=?, vec=? WHERE id=?", (len(vec), vec.tobytes(), nid))
            done += 1
        self.db.commit()
        return done

    # ---------------------------------------------------------------- read
    def search(self, query: str, k: int = 4, min_score: float = 0.5) -> list:
        rows = self.db.execute("SELECT id, ts, kind, text, dim, vec FROM notes").fetchall()
        if not rows or not (query or "").strip():
            return []
        qv = self._embed(query, query=True)
        qtok = _tokens(query)
        hits = []
        for nid, ts, kind, text, dim, blob in rows:
            if qv is not None and blob is not None and dim == len(qv):
                score = sum(map(mul, qv, self._load(blob)))            # cosine similarity (unit vectors)
            else:
                kw = _keyword_score(qtok, text)                        # fallback: weaker than a semantic match
                score = 0.35 + 0.5 * kw if kw else 0.0
            if score >= min_score:
                hits.append({"id": nid, "ts": ts, "kind": kind, "text": text, "score": round(score, 3)})
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:k]

    def recent(self, n: int = 15) -> list:
        return [{"id": r[0], "ts": r[1], "kind": r[2], "text": r[3]} for r in
                self.db.execute("SELECT id, ts, kind, text FROM notes ORDER BY id DESC LIMIT ?", (n,))]

    def get(self, note_id: int):
        r = self.db.execute("SELECT id, ts, kind, text FROM notes WHERE id=?", (int(note_id),)).fetchone()
        return {"id": r[0], "ts": r[1], "kind": r[2], "text": r[3]} if r else None

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]


if __name__ == "__main__":
    m = Memory(None)
    print(f"{m.count()} notes in {m.path}")
    for n in reversed(m.recent(50)):
        print(f"#{n['id']:<4} {n['ts'][:10]}  [{n['kind']}]  {n['text']}")
