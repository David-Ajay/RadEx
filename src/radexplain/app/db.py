"""Tiny SQLite layer: users, their analysed X-rays ("cases") and each case's chat messages.

Everything lives in one local file (data/app.db). Passwords are never stored as text: each gets a random
salt and is hashed with PBKDF2. All queries use ? placeholders and every case lookup checks the owner.
"""
import hashlib
import hmac
import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from radexplain.config import ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    username_lower TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cases(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    source_name TEXT,
    image BLOB,
    result_json TEXT NOT NULL,
    report_json TEXT NOT NULL,
    model_run TEXT,
    overlay BLOB,
    overlay_legend TEXT,
    created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS messages(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources_json TEXT,
    created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_cases_user ON cases(user_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_msgs_case ON messages(case_id, id);
"""
ITERATIONS = 200_000


def _now():
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect():
    path = ROOT / "data" / "app.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init():
    with connect() as con:
        con.executescript(SCHEMA)
        have = {r["name"] for r in con.execute("PRAGMA table_info(cases)")}      # upgrade databases made before overlays
        for col, kind in (("overlay", "BLOB"), ("overlay_legend", "TEXT")):
            if col not in have:
                con.execute(f"ALTER TABLE cases ADD COLUMN {col} {kind}")


def _hash(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), ITERATIONS).hex()


# ----------------------------------------------------------------------------- users
def register(username: str, password: str):
    """Returns (user, error). Usernames and passwords can be anything (just not empty)."""
    username = (username or "").strip()
    if not username:
        return None, "Please choose a username."
    if len(username) > 40:
        return None, "Username is too long (40 characters max)."
    if not password:
        return None, "Please choose a password."
    salt = secrets.token_hex(16)
    try:
        with connect() as con:
            cur = con.execute("INSERT INTO users(username, username_lower, password_hash, salt, created_at) "
                              "VALUES (?,?,?,?,?)", (username, username.lower(), _hash(password, salt), salt, _now()))
            return {"id": cur.lastrowid, "username": username}, None
    except sqlite3.IntegrityError:
        return None, "That username is already taken. Try another one, or log in."


def login(username: str, password: str):
    """Returns (user, error)."""
    with connect() as con:
        row = con.execute("SELECT * FROM users WHERE username_lower = ?", ((username or "").strip().lower(),)).fetchone()
    if row is None or not hmac.compare_digest(row["password_hash"], _hash(password or "", row["salt"])):
        return None, "Wrong username or password."
    return {"id": row["id"], "username": row["username"]}, None


# ----------------------------------------------------------------------------- cases
def create_case(user_id, title, source_name, image_bytes, result, report, model_run,
                overlay=None, overlay_legend=None) -> int:
    with connect() as con:
        cur = con.execute("INSERT INTO cases(user_id, title, source_name, image, result_json, report_json, model_run, "
                          "overlay, overlay_legend, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (user_id, title, source_name, image_bytes, json.dumps(result), json.dumps(report),
                           model_run, overlay, json.dumps(overlay_legend) if overlay_legend else None, _now()))
        return cur.lastrowid


def list_cases(user_id, limit=60) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT id, title, created_at FROM cases WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                           (user_id, limit)).fetchall()
    return [dict(r) for r in rows]


def get_case(user_id, case_id):
    """The case with its chat, or None if it doesn't exist or belongs to someone else."""
    with connect() as con:
        r = con.execute("SELECT * FROM cases WHERE id = ? AND user_id = ?", (case_id, user_id)).fetchone()
    if r is None:
        return None
    return {"id": r["id"], "title": r["title"], "source_name": r["source_name"], "image": r["image"],
            "result": json.loads(r["result_json"]), "report": json.loads(r["report_json"]),
            "model_run": r["model_run"], "created_at": r["created_at"], "overlay": r["overlay"],
            "overlay_legend": json.loads(r["overlay_legend"]) if r["overlay_legend"] else [],
            "messages": get_messages(user_id, case_id)}


def delete_case(user_id, case_id):
    with connect() as con:
        con.execute("DELETE FROM cases WHERE id = ? AND user_id = ?", (case_id, user_id))


# ----------------------------------------------------------------------------- chat
def add_message(user_id, case_id, role, content, sources=None) -> bool:
    with connect() as con:
        if con.execute("SELECT 1 FROM cases WHERE id = ? AND user_id = ?", (case_id, user_id)).fetchone() is None:
            return False
        con.execute("INSERT INTO messages(case_id, role, content, sources_json, created_at) VALUES (?,?,?,?,?)",
                    (case_id, role, content, json.dumps(sources or []), _now()))
        return True


def get_messages(user_id, case_id) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT m.role, m.content, m.sources_json FROM messages m JOIN cases c ON c.id = m.case_id "
                           "WHERE m.case_id = ? AND c.user_id = ? ORDER BY m.id", (case_id, user_id)).fetchall()
    return [{"role": r["role"], "content": r["content"], "sources": json.loads(r["sources_json"] or "[]")} for r in rows]
