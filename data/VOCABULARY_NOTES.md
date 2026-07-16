# Vocabulary selection notes

**Status: AVAILABILITY-CHECKED (2026-07-09), awaiting author sign-off.**
The full `python -m src.download --dry-run` sweep was run on 2026-07-09
(raw output in `data/availability_check_2026-07-09.txt`). 47/50 draft words
met the >= 3-variant criterion; three fell short and were substituted within
the same semantic category, per the methodology:

| Out (variants) | In (variants) | Category |
|----------------|---------------|----------|
| want (2)       | need (5)      | verbs    |
| come (2)       | ask (11)      | verbs    |
| tomorrow (2)   | morning (15)  | time     |

All 50 current words have between 4 and 28 variants. The list is frozen once
the author confirms it; any later change requires supervisor agreement and a
decision-log entry.

## Task framing

Isolated word-level BSL recognition over 50 classes, trained on pose-based
skeleton sequences (MediaPipe Holistic: pose + both hands + mouth region).
The vocabulary therefore favours signs whose identity is carried by manual
articulation (handshape, location, movement) that skeleton landmarks can
capture, supplemented by mouth-region landmarks.

## Selection criteria

1. **Everyday utility.** Words a beginner or an assistive lookup tool would
   plausibly need: greetings/courtesy, family, basic needs, common verbs,
   emotions, question words, time words.
2. **Manual distinctiveness.** Avoid sign pairs distinguished *only* by
   mouthing or by tiny handshape contrasts that image-plane landmarks resolve
   poorly (e.g. aunt/uncle style fingerspelling-initialised pairs).
   Known exception: *please* vs *thank-you* differ mainly in speed/emphasis
   of the same chin-outward movement; they are retained deliberately because
   they are core courtesy vocabulary and the feature set includes mouth
   landmarks and full temporal dynamics. Their confusion rate should be
   inspected in the confusion matrix.
3. **Handedness balance.** 30 one-handed / 20 two-handed signs, roughly
   reflecting the distribution in everyday BSL while exercising both hand
   channels of the model.
4. **Expected availability.** Words chosen to be common enough that
   SignBSL.com is likely to host >= 3 clips from multiple contributing
   organisations, enabling the organisation-grouped train/val split.
5. **URL form.** The `word` column uses SignBSL URL style: lowercase,
   hyphens for spaces (e.g. `thank-you`). Label names are these strings
   verbatim.

## Category breakdown (50 words)

| Category           | Count | Words |
|--------------------|-------|-------|
| greetings-courtesy | 8     | hello, goodbye, please, thank-you, sorry, yes, no, good |
| family             | 7     | mother, father, sister, brother, family, baby, friend |
| basic-needs        | 10    | eat, drink, water, toilet, help, home, sleep, work, school, money |
| verbs              | 10    | need, like, know, understand, think, go, ask, look, make, stop |
| emotions           | 6     | happy, sad, angry, love, tired, bad |
| question-words     | 5     | what, where, who, why, how |
| time               | 4     | today, morning, yesterday, week |

Handedness: 29 one-handed, 21 two-handed (after the 2026-07-09 substitutions).

## Known caveats

- BSL has substantial regional variation; the `notes` column flags words with
  well-known variants (mother, father, water, toilet, home, school, why).
  Clips of different regional variants under one label add intra-class
  variance; if a word's variants prove manually disjoint, prefer substitution.
- *today* was preferred over *now* to avoid a near-identical pair
  (BSL *today* is commonly a repeated *now*).
- Handedness labels are provisional descriptions of citation forms and should
  be verified against the downloaded SignBSL clips.

## Reserve list (substitution candidates)

In priority order: `book`, `house`, `milk`, `car`, `night`, `cold`, `hot`,
`big`, `small`, `name`.

Substitutions must preserve approximate category and handedness balance and
be re-checked with `python -m src.download --dry-run --words <word>`.

## Licensing

SignBSL.com aggregates videos from contributing organisations. Clips are
downloaded politely (>= 1 s between requests, identifying User-Agent) for
local academic research only, are never redistributed, and
`data/raw_videos/` is gitignored. Source URLs and organisations are recorded
per clip in `data/metadata.csv` for attribution.
