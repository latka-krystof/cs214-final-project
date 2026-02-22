"""
SmartSpec Benchmark — workload_benchmark.py

Tests two scenarios designed to reveal when speculative vs. standard decoding wins:

  SCENARIO 1 — "Speculative Wins"
      Low concurrency (1-2 users), long outputs, highly repetitive/structured text.
      The draft model can predict tokens well → high acceptance rate → big speedup.

  SCENARIO 2 — "Standard Wins"
      High concurrency (many parallel requests), short outputs, creative/diverse prompts.
      Draft model acceptance is low AND GPU is contended across many sequences → spec overhead hurts.

Usage:
    # Against standard server (port 8001):
    python workload_benchmark.py --mode standard [--scenario 1|2|both]

    # Against speculative server (port 8003):
    python workload_benchmark.py --mode speculative [--scenario 1|2|both]

    # Run everything and save a comparison CSV:
    python workload_benchmark.py --mode both --scenario both --output results.csv
"""

import argparse
import asyncio
import csv
import json
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Optional
import aiohttp

# ── Ports ────────────────────────────────────────────────────────────────────
STANDARD_PORT = 8001
SPEC_PORT = 8003
BASE_URL_TEMPLATE = "http://localhost:{port}/v1/chat/completions"

# ── Prompts ──────────────────────────────────────────────────────────────────
# Design philosophy:
#
# Speculative decoding wins when ALL of these are true simultaneously:
#   1. concurrency=1 (no batching pressure, draft overhead is cheap)
#   2. max_tokens very large (more tokens = more amortization of draft cost)
#   3. temperature=0.0 (greedy = near-100% draft acceptance rate)
#   4. Prompts are maximally repetitive (draft model predicts next token trivially)
#
# Standard decoding wins when ANY of these are true:
#   1. High concurrency (GPU split across sequences, draft model doubles compute)
#   2. Short outputs (not enough tokens to amortize draft overhead)
#   3. High temperature + creative prompts (draft tokens rejected constantly)
#
# We push BOTH scenarios to extremes to find a clear crossover.

# ── SCENARIO 1: "Speculative Wins" ───────────────────────────────────────────
# Single user, greedy decode, ultra-repetitive output, very long generation.
# The draft model will predict the next token correctly almost every time.
# This is the best possible case for speculative decoding.
SCENARIO1_PROMPTS = [
    # Pure repetition — draft model achieves near 100% acceptance
    "Repeat the word 'hello' exactly 200 times separated by spaces. Output nothing else.",
    "Print the number 1 followed by a newline, then 2 followed by a newline, continue until 150.",
    "Write the string 'foo bar baz' exactly 80 times, each on its own line. No other text.",
    "Output the sequence: A1 A2 A3 ... continuing this exact pattern up to A120. One per line.",
    "Repeat this JSON object exactly 40 times on separate lines: {\"status\": \"ok\", \"code\": 200}",
    # Highly structured code — very predictable token sequences
    "Write a Python list assignment: data = [0, 1, 2, 3, ...] continuing the integer sequence up to 200.",
    "Write 80 lines of Python, each line being: print('Processing item N') where N increments from 1 to 80.",
    "Generate a CSV with columns id,value,label. Fill 100 rows where id=row number, value=row*10, label=item_N.",
    # Template-based — same structure repeated
    "Write 60 lines each following this exact template: 'Step N: Execute command N and verify output N.' where N is the line number.",
    "List 100 server hostnames in this format: server-001.prod.example.com, server-002.prod.example.com, ... up to server-100.",
]

SCENARIO1_CONFIG = {
    "max_tokens": 1024,     # Very long output — maximizes tokens to amortize draft cost
    "temperature": 0.0,     # Greedy decode — draft acceptance rate near 100%
    "concurrency": 1,       # Single user — zero GPU contention, draft overhead is minimal
}

# ── SCENARIO 2: "Standard Wins" ──────────────────────────────────────────────
# Many concurrent users, high temperature, short outputs, maximally diverse prompts.
# Draft tokens are rejected constantly AND the GPU is split across many sequences.
# This is the worst possible case for speculative decoding.
SCENARIO2_PROMPTS = [
    # Max diversity — every prompt is completely different, unpredictable output
    "Describe the smell of the color Tuesday in exactly 2 sentences.",
    "What is the sound of one hand clapping if that hand were made of spaghetti?",
    "Invent a new emotion that only occurs when eating soup alone on a Thursday.",
    "Write a 2-sentence legal disclaimer for a time machine rental service.",
    "Describe quantum entanglement using only metaphors involving breakfast foods.",
    "What would Plato say about TikTok? Two sentences maximum.",
    "Invent a phobia name and its definition for the fear of slightly damp socks.",
    "Write a 2-sentence horoscope for someone born in the year of the WiFi router.",
    "Describe the geopolitics of an ant colony using Cold War terminology.",
    "What is the plot of a telenovela set inside a compiler? Two sentences.",
    "Invent a cocktail named after a sorting algorithm and describe it in 2 sentences.",
    "Write the opening of a nature documentary about office printers.",
    "Describe homesickness to someone who has always lived in the same place.",
    "What would a nihilist chef put on their menu? Two items with descriptions.",
    "Invent a sport that can only be played during a full moon. Two sentence rules.",
    "Describe the economy of a civilization that uses anxiety as currency.",
    "Write a 2-sentence weather forecast for a city that exists only in dreams.",
    "What life advice would a very old tortoise give to a mayfly? One sentence.",
    "Describe the taste of silence using only automotive terminology.",
    "Invent a new branch of mathematics that studies the behavior of lost socks.",
    "Write a 2-sentence pitch for an app that helps ghosts find WiFi networks.",
    "Describe photosynthesis as if it were a heated political debate.",
    "What would a GPS say if it had an existential crisis? Two sentences.",
    "Invent a holiday celebrated only by people who have never won anything.",
    "Write the terms and conditions for selling your shadow to a stranger.",
    "Describe the sensation of forgetting why you walked into a room, philosophically.",
    "What would a medieval knight think of a revolving door? Two sentences.",
    "Invent a new cardinal direction and explain what compass it points toward.",
    "Write a 2-sentence mission statement for a company that sells bottled silence.",
    "Describe the cultural significance of the left sock in a fictional society.",
]

SCENARIO2_CONFIG = {
    "max_tokens": 80,       # Short output — not enough tokens to amortize draft overhead
    "temperature": 1.2,     # Above 1.0 — maximally unpredictable, draft acceptance near 0%
    "concurrency": 32,      # Very high concurrency — GPU severely contended across sequences
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    prompt_idx: int
    success: bool
    ttft_ms: float = 0.0        # Time to first token (streaming)
    total_latency_ms: float = 0.0
    tokens_generated: int = 0
    throughput_tps: float = 0.0  # tokens/sec for this request
    error: str = ""


@dataclass
class ScenarioResult:
    scenario: int
    mode: str                   # "standard" or "speculative"
    port: int
    config: dict
    results: List[RequestResult] = field(default_factory=list)

    def successful(self):
        return [r for r in self.results if r.success]

    def summary(self):
        good = self.successful()
        if not good:
            return {"error": "all requests failed"}
        ttfts = [r.ttft_ms for r in good]
        lats = [r.total_latency_ms for r in good]
        tps_list = [r.throughput_tps for r in good]
        return {
            "scenario": self.scenario,
            "mode": self.mode,
            "total_requests": len(self.results),
            "successful": len(good),
            "concurrency": self.config["concurrency"],
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"],
            "ttft_p50_ms": round(statistics.median(ttfts), 1),
            "ttft_p95_ms": round(percentile(ttfts, 95), 1),
            "ttft_p99_ms": round(percentile(ttfts, 99), 1),
            "latency_p50_ms": round(statistics.median(lats), 1),
            "latency_p95_ms": round(percentile(lats, 95), 1),
            "latency_p99_ms": round(percentile(lats, 99), 1),
            "avg_throughput_tps": round(statistics.mean(tps_list), 2),
            "total_tokens": sum(r.tokens_generated for r in good),
        }


def percentile(data, p):
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (p / 100) * (len(sorted_data) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


# ── HTTP request with streaming (to measure TTFT) ────────────────────────────

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    prompt_idx: int,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    payload = {
        "model": "",          # vLLM ignores this when only one model is loaded
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,       # Streaming lets us capture TTFT
    }

    ttft_ms = 0.0
    total_latency_ms = 0.0
    tokens_generated = 0
    first_token_received = False

    async with semaphore:
        t_start = time.perf_counter()
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return RequestResult(
                        prompt_idx=prompt_idx,
                        success=False,
                        error=f"HTTP {resp.status}: {body[:200]}"
                    )

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")

                    if content and not first_token_received:
                        ttft_ms = (time.perf_counter() - t_start) * 1000
                        first_token_received = True

                    if content:
                        # Rough token count: split by spaces (good enough for timing purposes)
                        tokens_generated += max(1, len(content.split()))

            total_latency_ms = (time.perf_counter() - t_start) * 1000
            throughput_tps = (tokens_generated / (total_latency_ms / 1000)) if total_latency_ms > 0 else 0

            return RequestResult(
                prompt_idx=prompt_idx,
                success=True,
                ttft_ms=ttft_ms if first_token_received else total_latency_ms,
                total_latency_ms=total_latency_ms,
                tokens_generated=tokens_generated,
                throughput_tps=throughput_tps,
            )

        except Exception as e:
            return RequestResult(
                prompt_idx=prompt_idx,
                success=False,
                error=str(e),
                total_latency_ms=(time.perf_counter() - t_start) * 1000,
            )


# ── Scenario runner ───────────────────────────────────────────────────────────

async def run_scenario(
    scenario_num: int,
    mode: str,
    port: int,
    prompts: List[str],
    config: dict,
    repeats: int = 3,
) -> ScenarioResult:
    url = BASE_URL_TEMPLATE.format(port=port)
    concurrency = config["concurrency"]
    max_tokens = config["max_tokens"]
    temperature = config["temperature"]

    # Repeat the prompt list to get enough requests
    all_prompts = (prompts * repeats)[:max(len(prompts) * repeats, concurrency * 2)]

    semaphore = asyncio.Semaphore(concurrency)
    result = ScenarioResult(scenario=scenario_num, mode=mode, port=port, config=config)

    print(f"\n  → Sending {len(all_prompts)} requests | concurrency={concurrency} | max_tokens={max_tokens} | temp={temperature}")

    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            send_request(session, url, prompt, max_tokens, temperature, i, semaphore)
            for i, prompt in enumerate(all_prompts)
        ]
        raw_results = await asyncio.gather(*tasks)

    result.results = list(raw_results)
    failed = [r for r in raw_results if not r.success]
    if failed:
        print(f"  ⚠ {len(failed)} failed requests. First error: {failed[0].error}")

    return result


# ── Health check ──────────────────────────────────────────────────────────────

async def wait_for_server(port: int, timeout: int = 30) -> bool:
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(2)
    return False


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_summary(summaries: List[dict]):
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)

    # Group by scenario
    scenarios = sorted(set(s["scenario"] for s in summaries))
    for sc in scenarios:
        sc_name = "Scenario 1 — Speculative Expected to WIN (low concurrency, repetitive, long output)" if sc == 1 \
            else "Scenario 2 — Standard Expected to WIN (high concurrency, creative, short output)"
        print(f"\n{'─'*78}")
        print(f"  {sc_name}")
        print(f"{'─'*78}")
        print(f"  {'Metric':<28} {'STANDARD':>16} {'SPECULATIVE':>16}  {'Winner':>10}")
        print(f"  {'':─<28} {'':─>16} {'':─>16}  {'':─>10}")

        sc_sums = {s["mode"]: s for s in summaries if s["scenario"] == sc}
        std = sc_sums.get("standard", {})
        spc = sc_sums.get("speculative", {})

        def row(label, key, lower_is_better=True):
            sv = std.get(key, "N/A")
            spv = spc.get(key, "N/A")
            if isinstance(sv, float) and isinstance(spv, float):
                if lower_is_better:
                    winner = "STANDARD ✓" if sv < spv else "SPEC ✓" if spv < sv else "tie"
                else:
                    winner = "STANDARD ✓" if sv > spv else "SPEC ✓" if spv > sv else "tie"
                sv_str = f"{sv:.1f}"
                spv_str = f"{spv:.1f}"
            else:
                winner = ""
                sv_str = str(sv)
                spv_str = str(spv)
            print(f"  {label:<28} {sv_str:>16} {spv_str:>16}  {winner:>10}")

        row("TTFT p50 (ms)", "ttft_p50_ms")
        row("TTFT p95 (ms)", "ttft_p95_ms")
        row("TTFT p99 (ms)", "ttft_p99_ms")
        row("Latency p50 (ms)", "latency_p50_ms")
        row("Latency p95 (ms)", "latency_p95_ms")
        row("Avg throughput (tok/s)", "avg_throughput_tps", lower_is_better=False)
        row("Total tokens", "total_tokens", lower_is_better=False)

    print(f"\n{'=' * 80}\n")


def save_csv(summaries: List[dict], path: str):
    if not summaries:
        return
    import os
    file_exists = os.path.exists(path)
    keys = list(summaries[0].keys())
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(summaries)
    print(f"  Results appended to: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(args):
    modes_to_test = []
    if args.mode == "both":
        modes_to_test = [("standard", STANDARD_PORT), ("speculative", SPEC_PORT)]
    elif args.mode == "standard":
        modes_to_test = [("standard", STANDARD_PORT)]
    else:
        modes_to_test = [("speculative", SPEC_PORT)]

    scenarios_to_test = []
    if args.scenario == "both":
        scenarios_to_test = [1, 2]
    else:
        scenarios_to_test = [int(args.scenario)]

    # Verify servers are up
    for mode, port in modes_to_test:
        print(f"\nChecking {mode} server on port {port}...")
        ok = await wait_for_server(port, timeout=10)
        if not ok:
            print(f"  ✗ Server on port {port} is NOT reachable. Start it first with ./start_servers.sh")
            print(f"    Skipping {mode} tests.")
            modes_to_test = [(m, p) for m, p in modes_to_test if p != port]

    if not modes_to_test:
        print("No servers available. Exiting.")
        return

    all_summaries = []

    for scenario_num in scenarios_to_test:
        if scenario_num == 1:
            prompts = SCENARIO1_PROMPTS
            config = SCENARIO1_CONFIG
            scenario_label = "1 (repetitive/long/low-concurrency)"
        else:
            prompts = SCENARIO2_PROMPTS
            config = SCENARIO2_CONFIG
            scenario_label = "2 (creative/short/high-concurrency)"

        for mode, port in modes_to_test:
            print(f"\n{'='*60}")
            print(f"  Running Scenario {scenario_label}")
            print(f"  Mode: {mode.upper()}  |  Port: {port}")
            print(f"{'='*60}")

            # Warmup: 2 requests to load caches
            print("  [Warmup] Sending 2 warmup requests...")
            warmup_sem = asyncio.Semaphore(2)
            connector = aiohttp.TCPConnector(limit=4)
            async with aiohttp.ClientSession(connector=connector) as session:
                warmup_tasks = [
                    send_request(session, BASE_URL_TEMPLATE.format(port=port),
                                 prompts[0], config["max_tokens"], config["temperature"], -1, warmup_sem)
                    for _ in range(2)
                ]
                await asyncio.gather(*warmup_tasks)
            print("  [Warmup] Done.")
            await asyncio.sleep(1)

            result = await run_scenario(
                scenario_num=scenario_num,
                mode=mode,
                port=port,
                prompts=prompts,
                config=config,
                repeats=args.repeats,
            )

            summary = result.summary()
            all_summaries.append(summary)

            # Quick per-run print
            good = result.successful()
            if good:
                ttfts = [r.ttft_ms for r in good]
                lats = [r.total_latency_ms for r in good]
                print(f"\n  ✓ {len(good)}/{len(result.results)} succeeded")
                print(f"    TTFT   p50={statistics.median(ttfts):.0f}ms  p95={percentile(ttfts,95):.0f}ms")
                print(f"    Latency p50={statistics.median(lats):.0f}ms  p95={percentile(lats,95):.0f}ms")

    print_summary(all_summaries)

    if args.output:
        save_csv(all_summaries, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartSpec workload benchmark")
    parser.add_argument(
        "--mode",
        choices=["standard", "speculative", "both"],
        default="both",
        help="Which server(s) to benchmark (default: both — requires both ports up)"
    )
    parser.add_argument(
        "--scenario",
        choices=["1", "2", "both"],
        default="both",
        help="Which scenario to run: 1=spec-favored, 2=standard-favored (default: both)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="How many times to repeat the prompt list per scenario (more=more data, default=3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="smartspec_results.csv",
        help="CSV output file for results (default: smartspec_results.csv)"
    )
    args = parser.parse_args()
    asyncio.run(main(args))
