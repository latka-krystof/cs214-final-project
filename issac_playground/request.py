#!/usr/bin/env python3
"""
Replay vllm requests from a JSON trace file, firing each request
at its scheduled arrival_time relative to the start of the run.

Usage:
    python request.py --input dataset.json --model "meta-llama/Llama-3.2-3B-Instruct" --endpoint http://localhost:8000/v1/chat/completions 
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("replay")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    request_id: int
    arrival_time: float
    send_time: float          # wall-clock seconds after run start
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    output_text: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    phase: Optional[str] = None
    request_type: Optional[str] = None

    # Derived
    @property
    def latency(self) -> Optional[float]:
        if self.finish_time is not None:
            return self.finish_time - self.send_time
        return None

    @property
    def schedule_delay(self) -> float:
        """How late (seconds) we actually sent vs the target arrival_time."""
        return self.send_time - self.arrival_time


# ---------------------------------------------------------------------------
# Core sender
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    req: dict[str, Any],
    start_time: float,
    semaphore: asyncio.Semaphore,
    timeout: float,
) -> RequestResult:
    result = RequestResult(
        request_id=req["request_id"],
        arrival_time=req["arrival_time"],
        send_time=time.perf_counter() - start_time,
        phase=req.get("phase"),
        request_type=req.get("request_type"),
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": req["prompt"]}
        ],
        "max_tokens": req.get("max_tokens", 256),
        "temperature": req.get("temperature", 1.0),
        "stream": False,
    }

    async with semaphore:
        try:
            async with session.post(
                endpoint,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                result.status_code = resp.status
                body = await resp.json(content_type=None)

                if resp.status == 200:
                    choice = body["choices"][0]
                    result.output_text = choice.get("text") or choice.get("message", {}).get("content")
                    usage = body.get("usage", {})
                    result.prompt_tokens = usage.get("prompt_tokens")
                    result.completion_tokens = usage.get("completion_tokens")
                else:
                    result.error = json.dumps(body)

        except asyncio.TimeoutError:
            result.error = f"Timeout after {timeout}s"
        except Exception as exc:
            result.error = str(exc)

        result.finish_time = time.perf_counter() - start_time

    return result


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

async def replay(
    requests: list[dict],
    endpoint: str,
    model: str,
    concurrency: int,
    timeout: float,
) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []
    tasks: list[asyncio.Task] = []

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        start_time = time.perf_counter()
        log.info("Run started. Scheduling %d requests …", len(requests))

        for req in requests:
            target = req["arrival_time"]
            now = time.perf_counter() - start_time
            delay = target - now
            if delay > 0:
                await asyncio.sleep(delay)

            actual = time.perf_counter() - start_time
            log.info(
                "→ req %4d  target=%.3fs  actual=%.3fs  Δ=%+.3fs  phase=%-12s  %s",
                req["request_id"],
                target,
                actual,
                actual - target,
                req.get("phase", ""),
                req["prompt"][:60],
            )

            task = asyncio.create_task(
                send_request(session, endpoint, model, req, start_time, semaphore, timeout)
            )
            tasks.append(task)

        log.info("All %d requests dispatched. Waiting for completions …", len(tasks))
        results = await asyncio.gather(*tasks)

    return list(results)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_summary(results: list[RequestResult], total_wall: float) -> None:
    ok     = [r for r in results if r.status_code == 200]
    failed = [r for r in results if r.status_code != 200]
    latencies = [r.latency for r in ok if r.latency is not None]
    delays    = [r.schedule_delay for r in results]

    def pct(lst, p):
        if not lst:
            return float("nan")
        lst = sorted(lst)
        idx = int(len(lst) * p / 100)
        return lst[min(idx, len(lst) - 1)]

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  Total requests   : %d", len(results))
    log.info("  Successful (200) : %d", len(ok))
    log.info("  Failed           : %d", len(failed))
    log.info("  Wall time        : %.2f s", total_wall)
    log.info("  Throughput       : %.2f req/s", len(results) / total_wall)
    if latencies:
        log.info("  Latency (s)  p50=%.3f  p90=%.3f  p99=%.3f  max=%.3f",
                 pct(latencies, 50), pct(latencies, 90),
                 pct(latencies, 99), max(latencies))
    if delays:
        log.info("  Schedule delay (s)  mean=%.3f  max=%.3f",
                 sum(delays) / len(delays), max(delays))

    # Per-phase breakdown
    phases: dict[str, list] = {}
    for r in ok:
        phases.setdefault(r.phase or "unknown", []).append(r.latency)
    if phases:
        log.info("  Per-phase p50 latency:")
        for ph, lats in sorted(phases.items()):
            log.info("    %-20s  %d req  p50=%.3f s", ph, len(lats), pct(lats, 50))
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay a vllm JSON trace file.")
    p.add_argument("--input",       required=True,  help="Path to trace JSON file")
    p.add_argument("--endpoint",    required=True,  help="vllm /v1/completions URL")
    p.add_argument("--model",       required=True,  help="Model name served by vllm")
    p.add_argument("--output",      default=None,   help="Save per-request results to JSON")
    p.add_argument("--concurrency", type=int, default=64, help="Max in-flight requests (default 64)")
    p.add_argument("--timeout",     type=float, default=120.0, help="Per-request timeout in seconds")
    p.add_argument("--dry-run",     action="store_true", help="Print schedule without sending")
    p.add_argument("--verbose",     action="store_true", help="Enable DEBUG logging")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    if args.verbose:
        log.setLevel(logging.DEBUG)

    trace = json.loads(Path(args.input).read_text())
    requests = sorted(trace["requests"], key=lambda r: r["arrival_time"])
    log.info("Loaded %d requests from %s", len(requests), args.input)

    meta = trace.get("metadata", {})
    if meta:
        log.info(
            "Trace metadata: %d total req, duration=%.0fs, phases=%d",
            meta.get("total_requests", "?"),
            meta.get("duration_s", 0),
            len(meta.get("phases", [])),
        )

    if args.dry_run:
        log.info("DRY RUN — first 5 requests:")
        for r in requests[:5]:
            log.info("  t=%.3fs  req_id=%d  phase=%-12s  %s",
                     r["arrival_time"], r["request_id"],
                     r.get("phase", ""), r["prompt"][:60])
        return

    t0 = time.perf_counter()
    results = await replay(
        requests,
        endpoint=args.endpoint,
        model=args.model,
        concurrency=args.concurrency,
        timeout=args.timeout,
    )
    total_wall = time.perf_counter() - t0

    print_summary(results, total_wall)

    if args.output:
        out = []
        for r in results:
            d = asdict(r)
            d["latency"] = r.latency
            d["schedule_delay"] = r.schedule_delay
            out.append(d)
        Path(args.output).write_text(json.dumps(out, indent=2))
        log.info("Per-request results written to %s", args.output)


if __name__ == "__main__":
    asyncio.run(main())