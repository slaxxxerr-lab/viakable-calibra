"""Run all visual test suites and combine results.

Usage:
  python tests/visual/run_all.py
"""
import json
import subprocess
import sys
from pathlib import Path

SUITES = [
    Path(__file__).parent / "check_theme.py",
    Path(__file__).parent / "check_popover.py",
]
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def run_suite(path: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"Running: {path.name}")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=False,
    )
    report_name = "report_popover.json" if "popover" in path.name else "report.json"
    report_path = OUT / report_name
    if report_path.exists():
        data = json.loads(report_path.read_text(encoding="utf-8"))
        return {
            "suite": path.name,
            "exit_code": result.returncode,
            "total_failures": data.get("total_failures", len(data.get("failures", []))),
            "failures": data.get("failures", []),
        }
    return {
        "suite": path.name,
        "exit_code": result.returncode,
        "total_failures": -1 if result.returncode != 0 else 0,
        "failures": [],
    }


def main():
    all_results = []
    total_fails = 0

    for suite in SUITES:
        if not suite.exists():
            print(f"SKIP: {suite.name} not found")
            continue
        result = run_suite(suite)
        all_results.append(result)
        n = result["total_failures"]
        if n != 0:
            total_fails += max(n, 1)
        status = "PASS" if n == 0 else f"FAIL ({n})"
        print(f"\n{suite.name}: {status}")

    combined_path = OUT / "report_all.json"
    combined = {
        "total_failures": total_fails,
        "suites": all_results,
    }
    combined_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"ALL SUITES — total failures: {total_fails}")
    print("=" * 60)

    if total_fails > 0:
        for r in all_results:
            if r["total_failures"] != 0:
                print(f"\n  {r['suite']}: {r['total_failures']} failures")
                for f in r["failures"][:5]:
                    print(f"    {f}")
        sys.exit(1)
    else:
        print("ALL PASS")


if __name__ == "__main__":
    main()
