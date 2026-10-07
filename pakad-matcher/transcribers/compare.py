"""Every pitch source side by side, from the files its run wrote. Training/validation numbers only.

    poetry run python -m transcribers.compare          # -> transcribers/compare.md

Columns (definitions: DATA.md § Metrics):
  misread        reader vs notation, held out by recording (lower is better)
  phrase         per-samooha ranking score of the method chosen on validation (held-out estimate)
  directions     balanced accuracy of the chosen insight method, leave-one-clip-out
  nyas           nyas event F1 of the chosen insight method, leave-one-clip-out
"""

import json

import config as C

OUT = C.TRANSCRIBERS_DIR / "compare.md"


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def row(name, results):
    reader, s7 = _read(results / "reader.json"), _read(results / "s7_choice.json")
    ins = _read(results / "insights" / "choice.json")
    cell = lambda x: "—" if x is None else f"{x:.3f}"
    cv = (ins or {}).get("cv", {})
    pick = lambda task: (ins or {}).get(task, {}).get("choice")
    return [name, cell(reader and reader["cv_misread"]),
            cell(s7 and s7["val_auc"]) + (f" ({s7['chosen'].split(' (')[0]})" if s7 else ""),
            cell(cv.get(pick("directions"), {}).get("directions")) + (f" ({pick('directions')})" if ins else ""),
            cell(cv.get(pick("nyas"), {}).get("nyas_f1")) + (f" ({pick('nyas')})" if ins else "")]


def main():
    rows = [row("melodia", C.SHARED_RESULTS_DIR)]
    rows += [row(d.name, d / "results") for d in sorted(C.TRANSCRIBERS_DIR.iterdir())
             if (d / "results").is_dir()]
    head = ["source", "misread ↓", "phrase ↑ (method)", "directions ↑ (method)", "nyas F1 ↑ (method)"]
    md = "\n".join(["| " + " | ".join(r) + " |" for r in [head, ["---"] * len(head)] + rows])
    OUT.write_text("# Pitch sources compared (training/validation only)\n\n" + md + "\n")
    print(md)


if __name__ == "__main__":
    main()
