"""Punkt wejścia: liczy wynik obu torów i zapisuje SCORE.json.

    uv run python main.py                  # oba tory
    uv run python main.py --track classic  # tylko jeden tor (wynik drugiego zostaje w SCORE.json)

Rozwiązania torów: solutions/open.py i solutions/classic.py
(każdy eksportuje build_features, fit_predict, MODELS, SUMMARY).
"""

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path

from common.data import load_table
from common.evaluation import METRIC, evaluate, make_target
from common.tracks import TRACKS

AUTHOR = "oskar"  # ← ustaw swój nick (jak w nazwie brancha agent/<nick>)
AGENT = "claude-opus-5-5"       # ← model agenta AI, np. "claude-opus-5-5"
SCORE_PATH = Path("SCORE.json")


def run_track(track: str, matches):
    sol = importlib.import_module(f"solutions.{track}")
    X = sol.build_features(matches)
    y = make_target(matches)
    result = evaluate(sol.fit_predict, X, y, matches["season"])
    print(f"[{track:>7}] {result}")
    return {
        "cv_score": round(result.cv_score, 4),
        "cv_std": round(result.cv_std, 4),
        "holdout_score": round(result.holdout_score, 4),
        "folds": {k: round(v, 4) for k, v in result.folds.items()},
        "models": list(sol.MODELS),
        "summary": sol.SUMMARY,
        "updated": datetime.now().isoformat(timespec="seconds"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=TRACKS, help="uruchom tylko jeden tor")
    args = parser.parse_args()

    matches = load_table("Match").sort_values("date").reset_index(drop=True)

    score = json.loads(SCORE_PATH.read_text(encoding="utf-8")) if SCORE_PATH.exists() else {}
    score.update(author=AUTHOR, agent=AGENT, metric=METRIC, higher_is_better=True)
    score.setdefault("tracks", {})
    for track in [args.track] if args.track else TRACKS:
        score["tracks"][track] = run_track(track, matches)

    SCORE_PATH.write_text(json.dumps(score, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
