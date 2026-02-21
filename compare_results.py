"""
compare_results.py — merge two smartspec CSV runs into one side-by-side table.

Usage:
    python compare_results.py --standard standard_results.csv --speculative speculative_results.csv

Or if you ran both into the same CSV (appended):
    python compare_results.py --merged smartspec_results.csv
"""

import argparse
import csv
import statistics


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def coerce(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def print_comparison(all_rows):
    # Index by (scenario, mode)
    index = {}
    for row in all_rows:
        key = (int(row["scenario"]), row["mode"])
        index[key] = {k: coerce(v) for k, v in row.items()}

    scenarios = sorted(set(k[0] for k in index))

    METRICS = [
        ("TTFT p50 (ms)",            "ttft_p50_ms",        True),
        ("TTFT p95 (ms)",            "ttft_p95_ms",        True),
        ("TTFT p99 (ms)",            "ttft_p99_ms",        True),
        ("Latency p50 (ms)",         "latency_p50_ms",     True),
        ("Latency p95 (ms)",         "latency_p95_ms",     True),
        ("Latency p99 (ms)",         "latency_p99_ms",     True),
        ("Avg throughput (tok/s)",   "avg_throughput_tps", False),
        ("Total tokens",             "total_tokens",       False),
        ("Requests OK",              "successful",         False),
    ]

    SPEEDUP_METRICS = ["ttft_p50_ms", "latency_p50_ms", "avg_throughput_tps"]

    print("\n" + "=" * 82)
    print("  SMARTSPEC — SIDE-BY-SIDE COMPARISON")
    print("=" * 82)

    for sc in scenarios:
        sc_label = (
            "Scenario 1 — Spec expected to WIN  (low concurrency, repetitive, long output)"
            if sc == 1 else
            "Scenario 2 — Standard expected to WIN  (high concurrency, creative, short output)"
        )
        print(f"\n{'─'*82}")
        print(f"  {sc_label}")
        print(f"{'─'*82}")
        print(f"  {'Metric':<28} {'STANDARD':>14} {'SPECULATIVE':>14}  {'Δ vs Standard':>14}  {'Winner':>10}")
        print(f"  {'':─<28} {'':─>14} {'':─>14}  {'':─>14}  {'':─>10}")

        std = index.get((sc, "standard"), {})
        spc = index.get((sc, "speculative"), {})

        for label, key, lower_is_better in METRICS:
            sv = std.get(key, None)
            spv = spc.get(key, None)

            if isinstance(sv, float) and isinstance(spv, float):
                # Delta: positive means spec is worse for lower-is-better metrics
                if lower_is_better:
                    delta_pct = ((spv - sv) / sv * 100) if sv != 0 else 0
                    better = spv < sv
                    delta_str = f"{'+' if delta_pct > 0 else ''}{delta_pct:.1f}%"
                    winner = "STANDARD ✓" if sv < spv else ("SPEC ✓" if spv < sv else "tie")
                else:
                    delta_pct = ((spv - sv) / sv * 100) if sv != 0 else 0
                    delta_str = f"{'+' if delta_pct > 0 else ''}{delta_pct:.1f}%"
                    winner = "SPEC ✓" if spv > sv else ("STANDARD ✓" if sv > spv else "tie")

                sv_str  = f"{sv:>10.1f}"
                spv_str = f"{spv:>10.1f}"
            else:
                sv_str  = str(sv) if sv is not None else "N/A"
                spv_str = str(spv) if spv is not None else "N/A"
                delta_str = "—"
                winner = ""

            print(f"  {label:<28} {sv_str:>14} {spv_str:>14}  {delta_str:>14}  {winner:>10}")

        # Summary verdict
        if std and spc:
            ttft_winner  = "SPEC" if spc.get("ttft_p50_ms", 999) < std.get("ttft_p50_ms", 0) else "STANDARD"
            lat_winner   = "SPEC" if spc.get("latency_p50_ms", 999) < std.get("latency_p50_ms", 0) else "STANDARD"
            tput_winner  = "SPEC" if spc.get("avg_throughput_tps", 0) > std.get("avg_throughput_tps", 999) else "STANDARD"
            expected     = "SPEC" if sc == 1 else "STANDARD"
            print(f"\n  {'Verdict:':<12} TTFT winner={ttft_winner}  Latency winner={lat_winner}  Throughput winner={tput_winner}")
            print(f"  {'Expected:':<12} {expected} to win this scenario")

    print(f"\n{'=' * 82}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard",    help="CSV from standard-only run")
    parser.add_argument("--speculative", help="CSV from speculative-only run")
    parser.add_argument("--merged",      help="Single CSV containing both modes")
    args = parser.parse_args()

    if args.merged:
        rows = load_csv(args.merged)
    elif args.standard and args.speculative:
        rows = load_csv(args.standard) + load_csv(args.speculative)
    else:
        print("Provide either --merged <file>  OR  --standard <file> --speculative <file>")
        return

    print_comparison(rows)


if __name__ == "__main__":
    main()
