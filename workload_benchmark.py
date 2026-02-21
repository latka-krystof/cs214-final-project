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

# Scenario 1: Repetitive, structured, predictable → draft model loves these
SCENARIO1_PROMPTS = [
    "List the numbers from 1 to 50, one per line, with no extra text.",
    "Repeat the phrase 'The quick brown fox jumps over the lazy dog' exactly 10 times, each on its own line.",
    "Write a Python function that prints 'Hello, World!' and call it 10 times. Show all 10 print statements.",
    "List every month of the year followed by the number of days it has, one per line. Do not add any extra text.",
    "Count down from 30 to 1, one number per line, no other text.",
    "Write the multiplication table for 7 from 7x1 to 7x20, one equation per line.",
    "List the first 40 even numbers, one per line.",
    "Write a JSON array containing the strings 'apple', 'banana', 'cherry' repeated 15 times total.",
    "Print the ASCII alphabet in uppercase, one letter per line, 26 lines total.",
    "Write the phrase 'ERROR: connection timeout' exactly 20 times, one per line.",
]

SCENARIO1_CONFIG = {
    "max_tokens": 512,      # Long output → more tokens to speculate over
    "temperature": 0.0,     # Greedy / deterministic → highest draft acceptance
    "concurrency": 1,       # Low concurrency → no GPU contention
}

# Scenario 2: Creative, diverse, short → draft model struggles; high concurrency kills spec
SCENARIO2_PROMPTS = [
    "Give me a one-sentence creative tagline for a startup that sells underwater drones.",
    "In one sentence, what would a pirate say about machine learning?",
    "Write a haiku about debugging code at 3am.",
    "Suggest a quirky name for a coffee shop inside a library. One name only.",
    "In one sentence, describe the taste of nostalgia.",
    "What's an unusual hobby that pairs well with birdwatching? One sentence.",
    "Write the opening line of a sci-fi novel set on a sentient moon.",
    "Give one piece of life advice that sounds wrong but is actually right.",
    "Describe a color that doesn't exist using only smells.",
    "What would a robot's midlife crisis look like? One sentence.",
    "Invent a new word and define it in one sentence.",
    "Write a one-line tweet from a time-traveling accountant.",
    "Describe the sound of silence using only food metaphors. One sentence.",
    "What's the worst superpower to have in a library? One sentence answer.",
    "Give a motivational quote from a very pessimistic philosopher.",
    "Name one fictional sport that would be popular on Mars.",
    "What does déjà vu smell like? One sentence.",
    "Write the worst possible fortune cookie message.",
    "Describe gravity to someone who has never felt it. One sentence.",
    "What would a cat's resume look like? Give just the job title.",
]

SCENARIO2_CONFIG = {
    "max_tokens": 60,       # Short output → fewer tokens to amortize spec overhead
    "temperature": 1.0,     # High randomness → draft model acceptance plummets
    "concurrency": 16,      # High concurrency → GPU contention from running draft model
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
    with open(path, "a", newline="") as f:  # "a" instead of "w"
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
