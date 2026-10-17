"""Rerank caching + progressive /suggest/stream.

    cd backend && python3 test_suggest_cache.py

No network: Redis is disabled (in-process caches), the KG index, Neo4j and
the LLM rerank are stubbed. Exits non-zero on any failure, like the other
test_*.py scripts.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

os.environ["CLERK_AUTH_DISABLED"] = "1"
os.environ.pop("REDIS_URL", None)
sys.path.insert(0, os.path.dirname(__file__))

from src.config import load_config                        # noqa: E402
from src.recommender import Recommender, _Candidate       # noqa: E402
from src.schemas import SuggestRequest                    # noqa: E402

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  — {detail}"))
    if not ok:
        failures += 1


class FakeEmbedder:
    dimensions = 4

    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def make_recommender(llm_delay: float = 0.0):
    cfg = load_config()
    cfg.settings.cache.redis_url = None            # never touch the real Upstash
    rec = Recommender(cfg, neo=None, embedder=FakeEmbedder(), router=None)
    rec._loaded = True                             # skip the R2 / Neo4j index load
    calls = {"llm": 0, "fail": False}

    def fake_rank(emb, query):
        # Close scores (0.86 vs 0.85) → _should_run_rerank says yes, like the
        # real KG does for most queries.
        cands = [
            _Candidate(id=f"kg{i}", label=f"Variable {i}", category=None,
                       vector_score=0.86 - i * 0.01, raw_score=0.86 - i * 0.01,
                       alpha=0.86 - i * 0.01)
            for i in range(10)
        ]
        return "anchor", 0.9, cands

    def fake_llm(query, cands):
        calls["llm"] += 1
        time.sleep(llm_delay)
        if calls["fail"]:
            raise RuntimeError("LLM down")
        # Reverse the order so the rerank is visible in the output.
        for i, c in enumerate(cands):
            c.alpha = round(0.2 + 0.05 * i, 3)
            c.rationale = f"reranked {c.id}"

    rec._rank = fake_rank
    rec._llm_rerank = fake_llm
    rec._batch_edges_to_canvas = lambda ids, canvas: {}
    rec._propose_canvas_edges = lambda anchor, q, canvas: []
    return rec, calls


def request(label: str = "Level of pollution") -> SuggestRequest:
    return SuggestRequest(
        query_node={"id": "n1", "label": label},
        canvas={"nodes": [{"id": "n1", "label": label}], "edges": []},
    )


# ── 1. /suggest: the LLM runs once per query text, then replays ──────────
rec, calls = make_recommender()
r1 = rec.suggest(request())
r2 = rec.suggest(request())
check("suggest: LLM called once for two identical requests", calls["llm"] == 1, f"calls={calls['llm']}")
a1 = [(n.kg_id, n.alpha, n.rationale) for n in r1.new_nodes]
a2 = [(n.kg_id, n.alpha, n.rationale) for n in r2.new_nodes]
check("suggest: replayed response matches the reranked one", a1 == a2, f"{a1} != {a2}")
check("suggest: rerank actually applied", r2.new_nodes[0].rationale == "reranked kg9", str(a2[:2]))

rec.suggest(request("Car traffic"))
check("suggest: a different query text pays its own rerank", calls["llm"] == 2, f"calls={calls['llm']}")

# ── 2. stages(): vector then rerank; replay folds into the first event ───
rec, calls = make_recommender()
kinds = [k for k, _ in rec.stages(request())]
check("stages (cold): vector then rerank", kinds == ["vector", "rerank"], str(kinds))
stages = list(rec.stages(request()))
check("stages (cached): single vector event", [k for k, _ in stages] == ["vector"], str([k for k, _ in stages]))
check("stages (cached): no extra LLM call", calls["llm"] == 1, f"calls={calls['llm']}")
check("stages (cached): first event already reranked",
      stages[0][1].new_nodes[0].rationale == "reranked kg9", str(stages[0][1].new_nodes[:1]))

# The sync and streaming paths share one cache.
rec.suggest(request())
check("suggest after stages: still one LLM call", calls["llm"] == 1, f"calls={calls['llm']}")

# ── 3. A failed rerank is not cached ─────────────────────────────────────
rec, calls = make_recommender()
calls["fail"] = True
rec.suggest(request())
calls["fail"] = False
r = rec.suggest(request())
check("failed rerank retried on next request", calls["llm"] == 2, f"calls={calls['llm']}")
check("retry result is reranked", r.new_nodes[0].rationale == "reranked kg9", str(r.new_nodes[:1]))

# ── 4. /suggest/stream sends "vector" before the rerank finishes ─────────
# Driven at the ASGI level so we can timestamp each chunk as the app sends
# it — Starlette's TestClient buffers the whole response, which would hide
# whether the stream is actually progressive.
import anyio                                              # noqa: E402
from src import app as app_mod                            # noqa: E402

LLM_DELAY = 1.5
rec, calls = make_recommender(llm_delay=LLM_DELAY)
app_mod.app.state.recommender = rec
payload = request().model_dump_json().encode()


async def _stream_events():
    arrivals, status = {}, None
    t0 = time.perf_counter()
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await anyio.sleep(3600)          # client stays connected

    async def send(msg):
        nonlocal status
        if msg["type"] == "http.response.start":
            status = msg["status"]
        elif msg["type"] == "http.response.body":
            for line in msg.get("body", b"").decode().splitlines():
                if line.startswith("event:"):
                    arrivals.setdefault(line[6:].strip(), time.perf_counter() - t0)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/suggest/stream",
        "raw_path": b"/suggest/stream", "query_string": b"", "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"host", b"test")],
        "client": ("127.0.0.1", 1), "server": ("test", 80),
    }
    await app_mod.app(scope, receive, send)
    return status, arrivals, time.perf_counter() - t0


status, arrivals, _ = anyio.run(_stream_events)
check("stream: HTTP 200", status == 200, str(status))
order = sorted(arrivals, key=arrivals.get)
check("stream: events in order", order[:3] == ["thinking", "vector", "rerank"], str(order))
check(f"stream: vector arrives before the {LLM_DELAY}s rerank",
      arrivals.get("vector", 99) < LLM_DELAY * 0.5, f"{arrivals}")
check("stream: rerank arrives after the LLM delay",
      arrivals.get("rerank", 0) >= LLM_DELAY, f"{arrivals}")

# Second identical stream: replayed rerank, no LLM wait at all.
_, arrivals2, elapsed = anyio.run(_stream_events)
check("stream (cached): no rerank event, no LLM call",
      "rerank" not in arrivals2 and calls["llm"] == 1, f"{arrivals2}, calls={calls['llm']}")
check("stream (cached): fast", elapsed < LLM_DELAY * 0.5, f"{elapsed:.2f}s")

print("\nALL PASSED" if failures == 0 else f"\n{failures} FAILED")
sys.exit(1 if failures else 0)
