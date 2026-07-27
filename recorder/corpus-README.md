# G-Voice corpus — Polish TTS voice-cloning recording set

`corpus.json` — 593 sentences total to be read aloud one at a time and recorded.

## Counts

| source      | count |
|-------------|-------|
| `generated` | 585   |
| `wiki`      | 8     |
| **total**   | **593** |

IDs are sequential and zero-padded: `p0001` … `p0593`. `generated` sentences come first (`p0001`–`p0585`), `wiki` sentences last (`p0586`–`p0593`).

## Length mix (generated pool)

- short (4–7 words): 195 (~33%)
- medium (8–15 words): 346 (~59%, the dominant bucket, as requested)
- long (16–22 words): 44 (~8%)

## Sentence-type mix (whole corpus)

- questions (ends in `?`): 111
- exclamations (ends in `!`): 28
- statements / imperatives (ends in `.`): 454

## Phonetic-coverage approach

The generated pool was written by hand in topic-based batches (weather, family,
food, commute, hobbies, phone/assistant commands, numbers, etc.) rather than
templated, so sentences don't repeat structure — that matters for prosody
variety as much as for phoneme coverage. Coverage was tracked deliberately
across categories:

- **Digraphs/trigraphs and their soft/hard pairs** — dedicated batches drill
  `sz/ś-si`, `cz/ć-ci`, `rz/ż`, `dż`, `dź`, `zi/ź`, `ni/ń`, `l` vs `ł` in many
  word positions (onset, coda, across morpheme boundaries), not just in
  isolated example words.
- **Nasal vowels ą/ę** — a dedicated batch covers word-final `ę` (`zaczęła`,
  `wzięłam`), pre-consonant nasals, and the denasalization that happens before
  `ł`/`l` (`wzięli`, `wzięło`), plus ą/ę inside longer clauses so the model
  sees them in connected speech, not just citation form.
- **Hard consonant clusters** — a dedicated cluster/tongue-twister batch
  includes `wstrzymać`-, `źdźbło`-, `chrząszcz`-, `bezwzględny`-, `źle`-,
  `wszystko`-, `przez`-, `przyszłość`-style words and similar (`źdźbło`,
  `chrząszcz brzmi w trzcinie`, `rozszczepienie`, `wstrząs`, `zaszczepiliśmy`,
  `szczypta`, `wyszczerbiony`...). Most of the corpus stays naturally
  readable; only ~25–30 sentences lean into genuine tongue-twister density.
- **Sentence types** — statements, yes/no questions, wh-questions,
  exclamations and imperatives/commands were each written as their own
  batch so prosody (rising/falling intonation, command cadence) gets real
  coverage, not just declarative flatness.
- **Length variety** — after the first draft skewed too short (many
  questions/exclamations/commands are naturally 4–7 words), a rebalancing
  pass added ~250 more medium (8–15 word) and long (16–22 word) sentences,
  then capped the short bucket so medium sentences are clearly the majority,
  per the corpus spec.
- **Numbers/dates/quantities** — a dedicated batch spells out cardinal and
  ordinal numbers, times, dates, money and distances the way a Pole would
  actually say them aloud ("Dwudziestego drugiego lipca...", "Sto
  pięćdziesiąt trzy złote...", "Za dwadzieścia minut...", phone-number-style
  digit grouping, etc.) rather than leaving ambiguous digits in the text.
  Two digits pulled from `wiki` source material were likewise spelled out
  ("2" → "dwóch", "3D" → "trzy de") for unambiguous reading.
- **Register** — most sentences are warm, everyday, conversational
  (family life, weekend plans, commuting, cooking, "przypomnij mi...",
  "ustaw budzik...", personal-assistant-style commands) rather than
  encyclopedic prose. A small themed slice (~25–30 sentences, spread across
  other categories too) nods to the target speaker's actual interests —
  software/AI, rockets, 3D printing, piano — without dominating the corpus.
- **Safety** — no profanity, no offensive or embarrassing content, no
  sentence that is *only* a tongue twister with no natural-reading value,
  and no duplicates (verified programmatically).

## Wiki-sourced sentences — a note on scope

The spec asked for ~60–90 sentences pulled verbatim from
`~/Downloads/Claude/ClaudeMemory/wiki/*.md`. In practice only **8** genuine,
complete, self-contained Polish sentences could be pulled from that folder.

Why: of the 45 files in `wiki/`, 41 are technical/game-dev lesson notes
written in **English**, per this vault's own convention ("wiki techniczne
EN" — see `~/.claude/CLAUDE.md`). A character-level scan confirmed zero
Polish diacritics in all 41 of those files. Only three "seed" files
(`rocketry.md`, `druk-3d.md`, `muzyka.md`, migrated from an older `topics/`
folder and grandfathered as Polish) plus the `wiki/_index.md` table of
contents contain any Polish text at all — and most of that text is terse,
label-led bullet notes ("Sprzęt: Bambu Lab X1C Combo — z AMS...") rather
than grammatical, self-contained sentences, so most of it didn't qualify
under the "genuinely complete, grammatical sentence" rule either.

The 8 sentences that did qualify were extracted with only mechanical
normalization — stripping wiki-link brackets, markdown, and parenthetical
asides, replacing an em-dash with a comma in one case, and spelling out two
digits for unambiguous reading. No words were added, removed mid-sentence,
or reworded.

To keep the corpus at a healthy total size despite this shortfall, the
`generated` pool was expanded well past its own floor (585 sentences instead
of the ~550 minimum) rather than padding the `wiki` pool with fragments or
translated English content that wouldn't actually be "pulled" text.

If a larger wiki-flavored slice is wanted later, the options are: (a) relax
the wiki-only constraint to allow hand-translating a batch of the English
lesson notes into natural Polish (would need a new `source` tag, since that
content wouldn't be a verbatim pull), or (b) run `/harvest` on a few more
finished projects to grow the Polish-language footprint of the wiki over
time — neither was done here since it wasn't asked for.
