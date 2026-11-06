"""Ranking wszystkich branchy agent/* na podstawie ich SCORE.json — osobno dla każdego toru.

Użycie:
    uv run python scripts/leaderboard.py            # fetch + ranking
    uv run python scripts/leaderboard.py --no-fetch # bez git fetch
    uv run python scripts/leaderboard.py --by holdout_score
"""

import argparse
import json
import subprocess
import sys

REMOTE = "origin"
PREFIX = "agent/"
TRACKS = {"open": "DOWOLNY (open)", "classic": "KLASYCZNY (classic)"}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout


def fmt(v, w):
    return f"{v:>{w}.4f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--by", default="cv_score", choices=["cv_score", "holdout_score"])
    args = parser.parse_args()

    if not args.no_fetch:
        git("fetch", REMOTE, "--prune", "--quiet")

    refs = git(
        "for-each-ref",
        "--format=%(refname:short)|%(objectname:short)|%(committerdate:format:%Y-%m-%d %H:%M)",
        f"refs/remotes/{REMOTE}/{PREFIX}",
    ).splitlines()

    entries, missing = [], []
    for line in refs:
        ref, sha, date = line.split("|")
        branch = ref.removeprefix(f"{REMOTE}/")
        try:
            entries.append((branch, sha, date, json.loads(git("show", f"{ref}:SCORE.json"))))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            missing.append(branch)

    metrics = {e[3].get("metric") for e in entries}
    if len(metrics) > 1:
        print(f"UWAGA: różne metryki na branchach: {metrics} — wyniki nieporównywalne!\n")

    for track, title in TRACKS.items():
        rows = [(b, sha, d, s["tracks"][track]) for b, sha, d, s in entries
                if isinstance(s.get("tracks"), dict) and track in s["tracks"]]
        print(f"=== Tor {title} — sortowanie: {args.by} ===")
        if not rows:
            print("(brak wyników)\n")
            continue
        rows.sort(key=lambda r: r[3].get(args.by) if isinstance(r[3].get(args.by), (int, float)) else -1, reverse=True)
        header = f"{'#':>2}  {'branch':<22} {'cv_score':>9} {'±std':>7} {'holdout':>8}  {'commit':<8} {'ostatni commit':<16}  modele / opis"
        print(header)
        print("-" * len(header))
        for i, (branch, sha, date, t) in enumerate(rows, 1):
            models = ",".join(t.get("models", []))
            print(f"{i:>2}  {branch:<22} {fmt(t.get('cv_score'), 9)} {fmt(t.get('cv_std'), 7)} "
                  f"{fmt(t.get('holdout_score'), 8)}  {sha:<8} {date:<16}  [{models}] {t.get('summary', '')}")
        print()

    print("Metryka: balanced accuracy (wyżej = lepiej). Ranking: cv_score; werdykt końcowy: holdout_score.")
    if missing:
        print(f"Branche bez poprawnego SCORE.json: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
