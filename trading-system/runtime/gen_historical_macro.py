"""Generate 2019-2025 historical FOMC/CPI macro events as JSONL files.

This is critical: backtest uses 2019-onwards data, so MacroEventGate
needs historical FOMC/CPI events to trigger during backtest.

Output: /mnt/d/hermes/hermes_obsidian_vault/Hermes/MarketData/macro_calendar/YYYYMMDD.jsonl
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = Path("/mnt/d/hermes/hermes_obsidian_vault/Hermes/MarketData/macro_calendar")

# FOMC meeting announcement dates (UTC 19:00 = 2:00 PM ET)
# Source: Federal Reserve official calendar
FOMC_DATES = [
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020 (including emergency Mar 15)
    "2020-01-29", "2020-03-15", "2020-03-18", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022 (most aggressive hiking cycle)
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
]

# CPI release dates (UTC 13:30 = 8:30 AM ET)
# Source: BLS release calendar
CPI_DATES = [
    # 2019
    "2019-01-11", "2019-02-13", "2019-03-12", "2019-04-10",
    "2019-05-10", "2019-06-12", "2019-07-11", "2019-08-13",
    "2019-09-12", "2019-10-10", "2019-11-14", "2019-12-11",
    # 2020
    "2020-01-14", "2020-02-13", "2020-03-11", "2020-04-10",
    "2020-05-12", "2020-06-10", "2020-07-14", "2020-08-12",
    "2020-09-11", "2020-10-13", "2020-11-12", "2020-12-10",
    # 2021
    "2021-01-13", "2021-02-10", "2021-03-10", "2021-04-13",
    "2021-05-12", "2021-06-10", "2021-07-13", "2021-08-11",
    "2021-09-14", "2021-10-13", "2021-11-10", "2021-12-10",
    # 2022 (highest inflation in 40 years)
    "2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12",
    "2022-05-11", "2022-06-10", "2022-07-13", "2022-08-10",
    "2022-09-13", "2022-10-13", "2022-11-10", "2022-12-13",
    # 2023
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12",
    "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10",
    "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-13",
    # 2024
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
    "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-14", "2025-06-11", "2025-07-15",
]


def make_event(date_str: str, event_name: str, hour: int, minute: int) -> dict:
    """Build event record."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=hour, minute=minute, second=0, tzinfo=timezone.utc
    )
    ts = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "timestamp": ts,
        "event": event_name,
        "country": "US",
        "importance": "high",
        "actual": None,
        "forecast": None,
        "previous": None,
        "source": "historical_hardcoded",
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group events by date
    events_by_date = {}
    for d in FOMC_DATES:
        evt = make_event(d, "FOMC", 19, 0)  # 19:00 UTC = 2:00 PM ET
        date_compact = d.replace("-", "")
        events_by_date.setdefault(date_compact, []).append(evt)
    for d in CPI_DATES:
        evt = make_event(d, "CPI", 13, 30)  # 13:30 UTC = 8:30 AM ET
        date_compact = d.replace("-", "")
        events_by_date.setdefault(date_compact, []).append(evt)

    # Write JSONL files (do not overwrite existing files that may have actuals)
    new_count = 0
    updated_count = 0
    for date_compact, events in sorted(events_by_date.items()):
        fpath = OUTPUT_DIR / f"{date_compact}.jsonl"
        existing_records = []
        if fpath.exists():
            # Load existing and merge (preserve actuals/forecasts if present)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                existing_records.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            except OSError:
                pass

        # Merge: use existing if same timestamp+event, else append
        seen_keys = set()
        merged = []
        for r in existing_records:
            key = (r.get("timestamp", ""), r.get("event", ""))
            seen_keys.add(key)
            merged.append(r)
        for r in events:
            key = (r.get("timestamp", ""), r.get("event", ""))
            if key not in seen_keys:
                merged.append(r)
                seen_keys.add(key)

        # Deduplicate by (timestamp, event) keeping first
        seen = set()
        unique = []
        for r in merged:
            key = (r.get("timestamp", ""), r.get("event", ""))
            if key not in seen:
                seen.add(key)
                unique.append(r)

        with open(fpath, "w", encoding="utf-8") as f:
            for r in unique:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if existing_records:
            updated_count += 1
        else:
            new_count += 1

    # Summary
    total_events = sum(len(v) for v in events_by_date.values())
    print(f"[OK] Generated {total_events} events across {len(events_by_date)} dates")
    print(f"     New files: {new_count}, Updated files: {updated_count}")
    print(f"     FOMC events: {len(FOMC_DATES)}")
    print(f"     CPI events: {len(CPI_DATES)}")
    print(f"     Output dir: {OUTPUT_DIR}")

    # Verify by listing files
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jsonl")])
    print(f"\n[VERIFY] Total JSONL files in dir: {len(files)}")
    # Show year-by-year count
    by_year = {}
    for f in files:
        year = f[:4]
        by_year[year] = by_year.get(year, 0) + 1
    for y in sorted(by_year.keys()):
        print(f"  {y}: {by_year[y]} files")


if __name__ == "__main__":
    main()
