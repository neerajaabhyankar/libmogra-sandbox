"""Local annotation app: hear a candidate in context, watch the playhead, answer y/n.

    poetry run python annotate_app.py          # then open http://localhost:8765

Left is the pitch track of the surrounding musical sentence; the shaded span is the candidate
the matcher proposes, with its aligned path drawn on top. Keys: y / n / u, r replay,
space play-pause, ← → previous/next. Labels append to annotations/labels.jsonl as you go.
"""

import json
import re

import numpy as np
from datetime import datetime, timezone
from functools import lru_cache
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

import _bootstrap  # noqa: F401
import config as C
from utils import raagdb

LABELS = C.S3_DIR / "labels.jsonl"
PAGE = (C.HERE / "annotate_app.html").read_text
NOTATE_PAGE = (C.HERE / "notate_app.html").read_text


def chunks():
    return json.loads((C.S3_DIR / "chunks.json").read_text())


def notations():
    """Last notation per chunk wins."""
    out = {}
    if C.NOTATIONS.exists():
        for line in open(C.NOTATIONS):
            r = json.loads(line)
            out[r["chunk_id"]] = r
    return out


def decode_full(cents, swars, octaves, hop):
    """The alignment spanning the selection, with its rim absorbed, as one more candidate.

    Scored with `matcher.score_path`, the same re-score the free-ended candidates carry -- the
    decode's own per-frame cost is a different scale entirely, and mixing the two let a
    full-coverage alignment win on arithmetic rather than on fit.
    """
    import decode
    import matcher
    try:
        kinds, _per_frame = decode.align(cents, swars, hop, free_edges=True)
    except Exception:
        return None
    inside = np.flatnonzero(kinds != -2)                  # the rim is absorbed, not notated
    if not len(inside):
        return None
    f0, f1 = int(inside[0]), int(inside[-1])
    cost = matcher.score_path(cents[f0:f1 + 1], kinds[f0:f1 + 1], swars, octaves, hop, {**C.MATCH})[-1]
    return float(cost), f0, f1, kinds[f0:f1 + 1]


def chunk_contour(ch):
    import fullaudio
    ctr = fullaudio.contour(ch["video"])
    a, b = int(round(ch["t0"] / ctr.hop)), int(round(ch["t1"] / ctr.hop))
    return ctr.cents[a:b], ctr.hop


@lru_cache(maxsize=32)
def pool(slug):
    return json.loads((C.S3_DIR / "pool" / f"{slug}.json").read_text())


def pools():
    return sorted(p.stem for p in (C.S3_DIR / "pool").glob("*.json"))


def labels():
    if not LABELS.exists():
        return {}
    out = {}
    for line in open(LABELS):
        r = json.loads(line)
        if r.get("pool") == C.POOL_VERSION:
            out[(r["phrase_id"], r["index"])] = dict(verdict=r["verdict"],
                                                     comment=r.get("comment", ""))
    return out


class Handler(SimpleHTTPRequestHandler):
    def _send(self, body, ctype="application/json", code=200):
        body = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self._send(PAGE(), "text/html; charset=utf-8")
        if path == "/notate":
            return self._send(NOTATE_PAGE(), "text/html; charset=utf-8")
        if path == "/api/chunks":
            done = notations()
            return self._send(json.dumps([
                dict(**ch, segments=len(done.get(ch["id"], {}).get("segments", [])))
                for ch in chunks()]))
        if path.startswith("/api/chunk/"):
            cid = path.rsplit("/", 1)[1]
            ch = next(c for c in chunks() if c["id"] == cid)
            cents, hop = chunk_contour(ch)
            from utils import raagdb
            scale = sorted(raagdb.dataset_raags([ch["raag"]])[ch["raag"]].scale)
            prev = notations().get(cid, {})
            return self._send(json.dumps(dict(
                **ch, hop=hop, scale=scale, swar_names=raagdb.SWAR_NAMES,
                cents=[None if np.isnan(x) else round(float(x), 1) for x in cents],
                segments=prev.get("segments", []), note=prev.get("note", ""))))
        if path.startswith("/chunkaudio/"):
            f = C.CHUNK_DIR / path.rsplit("/", 1)[1]
            return self._send_audio(f) if f.exists() else self._send("{}", code=404)
        if path == "/api/phrases":
            done = labels()
            rows = []
            for slug in pools():
                d = pool(slug)
                n = len(d["items"])
                k = sum(1 for i in range(n) if (d["phrase_id"], i) in done)
                rows.append(dict(slug=slug, phrase_id=d["phrase_id"], phrase=d["phrase"],
                                 raag=d["raag"], source=d["source"], n=n, done=k))
            return self._send(json.dumps(rows))
        if path.startswith("/api/pool/"):
            d = dict(pool(path.rsplit("/", 1)[1]))
            d["swar_names"] = raagdb.SWAR_NAMES
            d["verdicts"] = {str(i): v for (pid, i), v in labels().items() if pid == d["phrase_id"]}
            return self._send(json.dumps(d))
        if path.startswith("/audio/"):
            f = C.S3_DIR / "audio" / path.rsplit("/", 1)[1]
            return self._send_audio(f) if f.exists() else self._send("{}", code=404)
        self._send("{}", code=404)

    def _send_audio(self, f):
        """Serve with byte ranges -- without them the browser cannot seek, so "play just the
        candidate" silently plays from the start."""
        data, n = f.read_bytes(), f.stat().st_size
        rng = self.headers.get("Range", "")
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip()) if rng else None
        if not m:
            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(n))
            self.end_headers()
            return self.wfile.write(data)
        start = int(m.group(1)) if m.group(1) else 0
        end = int(m.group(2)) if m.group(2) else n - 1
        if start >= n:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{n}")
            return self.end_headers()
        end = min(end, n - 1)
        chunk = data[start:end + 1]
        self.send_response(206)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Range", f"bytes {start}-{end}/{n}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/align":
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ch = next(c for c in chunks() if c["id"] == body["chunk_id"])
            cents, hop = chunk_contour(ch)
            import matcher
            from contour import Contour
            from utils import raagdb
            tokens = body["swars"].split()
            swars, octaves = raagdb.parse_phrase(tokens)
            if len(swars) != len(tokens) or len(swars) < 1:
                return self._send(json.dumps(dict(error="could not parse those swars")))
            # a selected sub-range, and free start/end inside it: silence, drone and anything
            # the notator did not write down are simply not covered
            i0 = max(0, int(round(body.get("t0", 0.0) / hop)))
            i1 = min(len(cents), int(round(body.get("t1", len(cents) * hop) / hop)))
            if i1 - i0 < 4:
                return self._send(json.dumps(dict(error="that range is too short")))
            cands = matcher.match(Contour(ch["id"], cents[i0:i1], hop), tuple(swars),
                                  top_k=8, octaves=tuple(octaves))
            if not cands:
                return self._send(json.dumps(dict(error="no alignment found in that range")))
            # a selection is an assertion that the sequence is *here*, so weigh covering it
            # against fitting it, rather than taking the tightest fit that happens to be cheapest
            voiced = ~np.isnan(cents[i0:i1])
            n_voiced = max(1, int(voiced.sum()))
            held = matcher._held(cents[i0:i1], hop, C.MATCH)
            full = decode_full(cents[i0:i1], tuple(swars), tuple(octaves), hop)

            def penalty(f0, f1, cost, cpath):
                cov = voiced[f0:f1 + 1].sum() / n_voiced
                on_note = cpath[: f1 + 1 - f0] >= 0
                sat = held[f0:f1 + 1][on_note]
                steady = sat.mean() if sat.size else 0.0      # notes should land on the notes
                return (cost + C.NOTATE_COVER_WEIGHT * (1.0 - cov)
                        + C.NOTATE_HELD_WEIGHT * (1.0 - steady)), cov
            options = [(penalty(x.f0, x.f1, x.cost, x.path)[0], x.f0, x.f1, x.cost, x.path)
                       for x in cands]
            if full:
                fc, ff0, ff1, fpath = full
                options.append((penalty(ff0, ff1, fc, fpath)[0], ff0, ff1, fc, fpath))
            _score, f0, f1, cost, cpath = min(options, key=lambda o: o[0])
            while f1 > f0 and not voiced[f0]:                 # don't draw notes over silence
                f0 += 1
                cpath = cpath[1:]
            while f1 > f0 and not voiced[f1]:
                f1 -= 1
                cpath = cpath[:-1]
            kinds = np.full(i1 - i0, -2, int)
            kinds[f0:f1 + 1] = cpath[: f1 + 1 - f0]
            cov = float(voiced[f0:f1 + 1].sum() / n_voiced)
            return self._send(json.dumps(dict(
                kinds=[int(k) for k in kinds], i0=i0, i1=i1, tokens=tokens,
                fit=round(float(cost), 3), coverage=round(cov, 2),
                t0=round((i0 + f0) * hop, 2), t1=round((i0 + f1 + 1) * hop, 2))))
        if path == "/api/notation":
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ch = next(c for c in chunks() if c["id"] == body["chunk_id"])
            rec = dict(chunk_id=ch["id"], raag=ch["raag"], kind=ch["kind"], video=ch["video"],
                       t0=ch["t0"], t1=ch["t1"], tonic_hz=ch["tonic_hz"],
                       segments=body.get("segments", []), note=(body.get("note") or "").strip(),
                       annotator=body.get("who", "neeraja"), matcher=C.MATCHER_VERSION,
                       ts=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            C.S3_DIR.mkdir(parents=True, exist_ok=True)
            with open(C.NOTATIONS, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            return self._send(json.dumps(dict(ok=True)))
        if path != "/api/label":
            return self._send("{}", code=404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        d = pool(body["slug"])
        item = d["items"][body["index"]]
        rec = dict(phrase_id=d["phrase_id"], phrase=d["phrase"], raag=d["raag"],
                   index=body["index"], verdict=body["verdict"],
                   comment=(body.get("comment") or "").strip(),
                   annotator=body.get("who", "neeraja"),
                   matcher=d["matcher"], pool=d["pool"],
                   ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   **{k: item[k] for k in ("video", "t0", "t1", "dur", "win_t0", "win_t1",
                                           "cost", "pitch_cost", "orn_frac", "leaps", "rank")})
        C.S3_DIR.mkdir(parents=True, exist_ok=True)
        with open(LABELS, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self._send(json.dumps(dict(ok=True)))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"phrases: http://localhost:{C.APP_PORT}    notation: http://localhost:{C.APP_PORT}/notate")
    ThreadingHTTPServer(("127.0.0.1", C.APP_PORT), Handler).serve_forever()
