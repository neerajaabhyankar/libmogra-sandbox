# notator — intentions for the notation tool

The notation view (`/notate`, `notate_app.html` + the endpoints in `annotate_app.py`) was built to
feed S5b: a corpus of human-heard swar sequences to measure the automatic reading against. But
Neeraja's note on 2026-09-23 is that it is worth more than that errand — *"after this exercise is
done, I'd like to further develop this notation tool — there's so much I could do with it!"*

This file is the parking lot for that. It is not a plan with stages; it is what we thought of while
building, so we don't have to think of it again.

---

## What it does today

Pick a chunk → see its pitch track with the swar grid → **drag to select a stretch** → type or click
the swars you hear → **align** places them on the contour and reports the fit and how much of the
selection it covers → **add stretch** keeps it. Stretches are listed under the plot, click one to
re-do it, `×` to drop it. Everything saves to `annotations/notations.jsonl` as a list of stretches
per chunk, last save wins.

Design decisions already made, and why:

- **Coverage counts, but the notes have to land on the notes.** Alignment has free ends inside the
  selection (chunks contain silence and drone nobody writes down), and a selection is an assertion
  that the sequence is *here*, so candidates are scored
  `cost + NOTATE_COVER_WEIGHT x (1 - coverage) + NOTATE_HELD_WEIGHT x (1 - on-held)` -- the last
  term is Neeraja's rule that a swar should sit on held pitch, not on the way to it. The
  span-covering alignment runs with **absorbing states at each rim**, so whatever the notator is
  not accounting for at the edges costs the same as unexplained ornament rather than dragging a
  note out to meet it. Coverage is always reported.
  *(Two bugs lived here: weighting coverage at 1.0 with no on-held term put swars at the extremes,
  and the span-covering candidate was being scored by the decode's own per-frame cost while the
  others carried the matcher re-score -- a different scale entirely, so it won on arithmetic. Both
  are why a `g` once landed on `` `S ``, 305 cents off, with a "fit" of 0.255.)*
- **A single swar is a legal notation** — the case that matters is exactly when the alignment fails
  and you want to pin one note down.
- **Sub-ranges, not one notation per chunk.** A 30-note taan gets split into 4–5 stretches "for
  convenience + clarity + correction where your alignment is wrong". Correction happens by
  re-selecting a smaller range, not by dragging individual notes.
- **A saved stretch stays editable.** Click its row to reopen it: it re-aligns as a live preview,
  the button becomes *update stretch*, and its swars or its edges can both change. `esc` leaves it
  alone, `×` removes it. Nothing reaches disk until *save chunk*, and the status line says when
  there is something unsaved.
- **A swar keypad**, `,P` to `` `P ``, laid out like a keyboard: every natural gets the same step,
  komal and teevra M sit between their neighbours, and the saptaks are shown by a band behind
  madhya rather than by gaps or rules — so `G m` and `` N `S `` are spaced identically. All circles
  look the same; the height difference already says which are komal/teevra. Clicking appends.
- **Raag notes can be marked by hand** (`mark raag notes`, then click circles): a ring on the
  keypad and an emphasised grid line in the plot, remembered per raag. It is a **visual aid the
  notator sets**, never inferred for them — nothing about it feeds the model.
- **Colour means state, not judgment**: a stretch being aligned is blue, a stretch already added is
  green on its green band, and ornament/transit is purple — it is a category, not a fault.
- **The selection has draggable edges** (`ew-resize` near either end), and re-aligns as you pull
  it, so a stretch can be grown into rather than redrawn.
- **Playback never fights typing.** `⇧space` play/pause and `⇧K` play-selection work mid-word;
  plain `space` and `p` still work when the cursor is not in a text box. Playing a selection runs
  it **once** and stops at its end, rather than looping or running on.
- **No flags at all.** "Has non-voice pitch" went because drone and stray pitch are always there;
  "unclear" went because a stretch you cannot hear is one you do not notate.

## Ideas, roughly in the order they would pay off

**Notating**
- Per-note correction: drag a swar's boundary or reassign it, for when the alignment is close but
  wrong in one place. Deliberately deferred — sub-ranges may make it unnecessary.
- Keyboard entry for the keypad (`s r R g G m M p d D n N`, shift for taar, alt for mandra) so the
  hands never leave the keys.
- Marked raag notes could do more than decorate: grey out what is outside them, or count how often
  the notation strays from them — the first statistic the tool could answer about itself.
- Play a stretch on loop while typing, and slow it further (0.25×) for dense taans.
- A "same again" button: copy the previous stretch's swars, for repeated phrases in a bandish.
- Audible feedback: play the notated sequence back as tones against the audio, to check by ear.

**Seeing**
- Overlay the *matcher's* reading next to the human's, so disagreements are visible while notating
  (this is the S6 evaluation, but live).
- Mark nyas and phrase boundaries (a breath, a sam) as light annotations on top of the swars.
- Show the tanpura's Sa/Pa lines distinctly from the sung line, once we can tell them apart.

**Beyond one chunk**
- Notate a whole recording in passes, not fixed chunks: scroll through, notate what is clear, leave
  the rest.
- Import an existing notation (a bandish someone wrote) and align it to a recording — the same
  alignment machinery, run at scale. This is how a notated corpus could grow without notating.
- Export: a notated recording as a swar sequence with times, for the statistical query layer, and
  as something printable in Bhatkhande notation.
- Multiple annotators on the same chunk, to measure how much two musicians disagree. That number
  caps every accuracy we report and we currently have no estimate of it.

**Wider than this project**
- A teaching/practice view: sing into the mic, see your contour against a reference notation.
- Searching a personal library: "find me every place this artist sings this phrase", which is the
  `pakad.py` tool with a notated corpus behind it.

## Things to watch

- The tool shapes the data. If alignment is wrong in a systematic way and correcting it is tedious,
  the corpus will quietly inherit that bias. Keep the raw typed sequence, the alignment, and the
  fit separately (we do) so this can be checked later.
- Notating is slow and the corpus will stay small. Design every use of it for tens of chunks, not
  thousands, and prefer methods that can be checked by eye.
