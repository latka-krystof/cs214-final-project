"""
simple_filter.py

Same as workload_benchmark.py but with ONE combined dataset.
A filter function looks at each request and routes it to:
  - Port 8003 (speculative) if max_tokens >= TOKEN_THRESHOLD
  - Port 8001 (standard)    if max_tokens <  TOKEN_THRESHOLD

Usage:
    python simple_filter.py
"""

import asyncio
import aiohttp
import json
import time
import statistics
from dataclasses import dataclass, field
from typing import List

STANDARD_PORT    = 8001
SPECULATIVE_PORT = 8003
TOKEN_THRESHOLD  = 200   # requests above this go to speculative, below to standard

# ── Combined dataset ──────────────────────────────────────────────────────────
# Each entry is (prompt, max_tokens, temperature)
# Long/repetitive requests → speculative
# Short/creative requests  → standard

REQUESTS = [
    # ── Long, repetitive → should route to SPECULATIVE ────────────────────────
    ("Repeat the word 'hello' exactly 200 times separated by spaces. Output nothing else.",                      512, 0.0),
    ("Print the number 1 followed by a newline, then 2, continue until 150. One per line.",                     512, 0.0),
    ("Write the string 'foo bar baz' exactly 80 times, each on its own line. No other text.",                   512, 0.0),
    ("List the first 100 even numbers, one per line, no other text.",                                           512, 0.0),
    ("Write 60 lines each: 'Step N: Execute command N.' where N is the line number.",                           512, 0.0),
    ("Generate a CSV with columns id,value. Fill 80 rows where id=row number and value=row*10.",                400, 0.0),
    ("Write the multiplication table for 7 from 7x1 to 7x30, one equation per line.",                          400, 0.0),
    ("Repeat this JSON object 40 times on separate lines: {\"status\": \"ok\", \"code\": 200}",                 400, 0.0),
    ("List every integer from 50 down to 1, one per line, no other text.",                                      300, 0.0),
    ("Write 50 lines of Python, each: print('Processing item N') where N goes from 1 to 50.",                  400, 0.0),

    # ── Short, creative → should route to STANDARD ────────────────────────────
    ("Give me a one-sentence tagline for a startup that sells underwater drones.",                               60,  1.0),
    ("In one sentence, what would a pirate say about machine learning?",                                         60,  1.0),
    ("Write a haiku about debugging code at 3am.",                                                               40,  1.0),
    ("Suggest a quirky name for a coffee shop inside a library. One name only.",                                 20,  1.0),
    ("In one sentence, describe the taste of nostalgia.",                                                        50,  1.0),
    ("What would a robot's midlife crisis look like? One sentence.",                                             60,  1.0),
    ("Invent a new word and define it in one sentence.",                                                         50,  1.0),
    ("Write the worst possible fortune cookie message.",                                                         30,  1.0),
    ("What is the worst superpower to have in a library? One sentence.",                                         50,  1.0),
    ("Describe the sound of silence using only food metaphors. One sentence.",                                   60,  1.0),
]


# ── Routing filter ────────────────────────────────────────────────────────────

def route(max_tokens: int) -> tuple:
    """Returns (backend_name, port) based on max_tokens."""
    if max_tokens >= TOKEN_THRESHOLD:
        return "speculative", SPECULATIVE_PORT
    else:
        return "standard", STANDARD_PORT


# ── Result tracking ───────────────────────────────────────────────────────────

@dataclass
class Result:
    prompt:     str
    max_tokens: int
    backend:    str
    success:    bool
    ttft_ms:    float = 0.0
    latency_ms: float = 0.0
    tokens:     int   = 0
    error:      str   = ""


# ── Send one request ──────────────────────────────────────────────────────────

async def send(session, prompt, max_tokens, temperature, port, sem):
    url     = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model":       "",
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      True,
    }
    ttft_ms = 0.0
    tokens  = 0
    first   = False

    async with sem:
        t0 = time.perf_counter()
        try:
            async with session.post(url, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return {"success": False, "error": f"HTTP {resp.status}: {body[:100]}",
                            "ttft_ms": 0, "latency_ms": 0, "tokens": 0}

                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk   = json.loads(data)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content and not first:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                            first   = True
                        if content:
                            tokens += max(1, len(content.split()))
                    except Exception:
                        continue

            latency_ms = (time.perf_counter() - t0) * 1000
            return {"success": True, "ttft_ms": ttft_ms or latency_ms,
                    "latency_ms": latency_ms, "tokens": tokens, "error": ""}

        except Exception as e:
            return {"success": False, "error": str(e),
                    "ttft_ms": 0,
                    "latency_ms": (time.perf_counter() - t0) * 1000,
                    "tokens": 0}


# ── Run benchmark ─────────────────────────────────────────────────────────────

async def run(repeats=3):
    all_requests = REQUESTS * repeats
    sem          = asyncio.Semaphore(8)

    spec_count = len([r for r in all_requests if r[1] >= TOKEN_THRESHOLD])
    std_count  = len([r for r in all_requests if r[1] <  TOKEN_THRESHOLD])
    print(f"\n  Sending {len(all_requests)} total requests")
    print(f"  Filter:  {spec_count} → speculative (max_tokens >= {TOKEN_THRESHOLD})")
    print(f"           {std_count} → standard    (max_tokens <  {TOKEN_THRESHOLD})\n")

    connector = aiohttp.TCPConnector(limit=16)
    results   = []

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for prompt, max_tokens, temperature in all_requests:
            backend, port = route(max_tokens)
            tasks.append((prompt, max_tokens, backend,
                          send(session, prompt, max_tokens, temperature, port, sem)))

        raw = await asyncio.gather(*[t[3] for t in tasks])

        for (prompt, max_tokens, backend, _), r in zip(tasks, raw):
            results.append(Result(
                prompt     = prompt[:60],
                max_tokens = max_tokens,
                backend    = backend,
                success    = r["success"],
                ttft_ms    = r["ttft_ms"],
                latency_ms = r["latency_ms"],
                tokens     = r["tokens"],
                error      = r["error"],
            ))

    return results


# ── Print summary ─────────────────────────────────────────────────────────────

def percentile(data, p):
    if not data:
        return 0.0
    s   = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lo  = int(idx)
    hi  = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def summarize(results):
    for backend in ["speculative", "standard"]:
        group = [r for r in results if r.backend == backend and r.success]
        fail  = [r for r in results if r.backend == backend and not r.success]

        print(f"\n{'─'*58}")
        print(f"  {backend.upper()}  (port {'8003' if backend == 'speculative' else '8001'})")
        print(f"  Routed:  max_tokens {'≥' if backend == 'speculative' else '<'} {TOKEN_THRESHOLD}")
        print(f"{'─'*58}")

        if not group:
            print(f"  ✗ All {len(fail)} requests failed.")
            if fail:
                print(f"    First error: {fail[0].error}")
            continue

        ttfts = [r.ttft_ms    for r in group]
        lats  = [r.latency_ms for r in group]
        toks  = [r.tokens     for r in group]
        tps   = [r.tokens / (r.latency_ms / 1000) for r in group if r.latency_ms > 0]

        print(f"  Requests : {len(group)} ok, {len(fail)} failed")
        print(f"  TTFT     : p50={statistics.median(ttfts):.0f}ms  p95={percentile(ttfts,95):.0f}ms  p99={percentile(ttfts,99):.0f}ms")
        print(f"  Latency  : p50={statistics.median(lats):.0f}ms   p95={percentile(lats,95):.0f}ms   p99={percentile(lats,99):.0f}ms")
        print(f"  Tokens   : avg={statistics.mean(toks):.0f}  total={sum(toks)}")
        if tps:
            print(f"  Thruput  : avg={statistics.mean(tps):.1f} tok/s")

    print(f"\n{'─'*58}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    print("\n" + "=" * 58)
    print("  SmartSpec — Simple Filter Benchmark")
    print(f"  Routing: max_tokens >= {TOKEN_THRESHOLD} → speculative (:8003)")
    print(f"           max_tokens <  {TOKEN_THRESHOLD} → standard    (:8001)")
    print("=" * 58)

    # Warmup
    print("\n  [Warmup] 1 request per backend...")
    sem = asyncio.Semaphore(2)
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            send(session, "Say hi.", 10, 0.0, STANDARD_PORT,    sem),
            send(session, "Say hi.", 10, 0.0, SPECULATIVE_PORT, sem),
        )
    print("  [Warmup] Done.")

    results = await run(repeats=3)
    summarize(results)


if __name__ == "__main__":
    asyncio.run(main())
