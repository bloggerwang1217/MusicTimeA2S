# Italian tempo marking → quarter-note BPM lookup.
#
# Three tiers of sourcing:
#
# 1. MuseScore 4 palette defaults — the 13 standard Italian tempo markings
#    hardcoded in MuseScore's UI palette. Used for slow-to-moderate terms
#    (Grave through Allegretto) where MuseScore values closely match our
#    empirical data.
#    Source: src/palette/internal/palettecreator.cpp, newTempoPalette()
#    https://github.com/musescore/MuseScore
#
# 2. music21.tempo.defaultTempoValues — MIT music21 library (Cuthbert Lab),
#    values based on Harvard Dictionary of Music conventions. Used for
#    fast terms (Allegro+) where MuseScore's values are higher than
#    empirical medians, and for terms not in MuseScore's palette.
#
# 3. Zeng et al. — https://github.com/wei-zeng98/piano-a2s (Apache 2.0).
#    Compound markings (poco/molto/assai variants) from their codebase.
#    Zeng's slow-term values closely match MuseScore's (likely derived
#    from the same source).
#
# Grove Dictionary: Vivo = Vivace; Vivacissimamente ≈ Vivacissimo.

from __future__ import annotations

import re
from typing import Optional

from music21.tempo import defaultTempoValues as _M21_TEMPOS

# --- Tier 1: MuseScore 4 palette defaults (slow–moderate) ---
_MUSESCORE_SLOW = {
    "grave": 35,
    "largo": 50,
    "lento": 52,
    "larghetto": 63,
    "adagio": 71,
    "andante": 92,
    "andantino": 94,
    "moderato": 114,
    "allegretto": 116,
}

# Start from music21 (broadest coverage), then override slow–moderate
# with MuseScore. Fast terms (allegro+) keep music21 values.
CLASSIC_TEMPOS: dict[str, int] = dict(_M21_TEMPOS)
CLASSIC_TEMPOS.update(_MUSESCORE_SLOW)

# --- Tier 3: Zeng et al. compound markings ---
CLASSIC_TEMPOS["largoassai"] = 40       # Zeng
CLASSIC_TEMPOS["pocolargo"] = 60        # Zeng
CLASSIC_TEMPOS["pocoadagio"] = 76       # Zeng
CLASSIC_TEMPOS["menuetto"] = 112        # Zeng
CLASSIC_TEMPOS["pocoallegretto"] = 116  # Zeng
CLASSIC_TEMPOS["pocoallegro"] = 124     # Zeng
CLASSIC_TEMPOS["allegroassai"] = 138    # Zeng
CLASSIC_TEMPOS["allegrovivace"] = 160   # Zeng
CLASSIC_TEMPOS["allegrovivaceassai"] = 170  # Zeng
CLASSIC_TEMPOS["pocopresto"] = 180      # Zeng
CLASSIC_TEMPOS["prestoassai"] = 200     # Zeng

# vivace assai: Zeng's 150 conflicts with music21 vivace (160).
# "assai" ≈ "very" ≈ vivacissimo; align to vivacissimo (168).
CLASSIC_TEMPOS["vivaceassai"] = CLASSIC_TEMPOS["vivacissimo"]  # 168

# Spaceless aliases for music21 compound terms.
CLASSIC_TEMPOS["allegromoderato"] = CLASSIC_TEMPOS["allegro moderato"]
CLASSIC_TEMPOS["moltoallegro"] = CLASSIC_TEMPOS["molto allegro"]

# Grove Dictionary.
CLASSIC_TEMPOS["vivo"] = CLASSIC_TEMPOS["vivace"]
CLASSIC_TEMPOS["vivacissimamente"] = CLASSIC_TEMPOS["vivacissimo"]

# "minuetto" (standard Italian) = "menuetto" (French-influenced, used by Zeng).
CLASSIC_TEMPOS["minuetto"] = CLASSIC_TEMPOS["menuetto"]


# --- Beat-unit string → quarter-note multiplier ---
_BEAT_UNITS = {
    "whole": 4.0,
    "half-dot": 3.0, "dotted-half": 3.0, "half dot": 3.0,
    "half note": 2.0, "half-note": 2.0, "half": 2.0,
    "quarter-dot": 1.5, "dotted-quarter": 1.5, "quarter dot": 1.5,
    "quarter": 1.0,
    "eighth": 0.5,
}

_RE_METRIC = re.compile(
    r"\[([a-z][a-z \-]+?)\]"
    r"\s*=?\s*"
    r"(\d+\.?\d*)",
    re.IGNORECASE,
)

_RE_MM = re.compile(
    r"(?:M\.?\s*M\.?|Metr):?\s*"
    r"(?:\[([a-z][a-z \-]+?)\])?"
    r"\s*=?\s*"
    r"(\d+\.?\d*)",
    re.IGNORECASE,
)


# Common abbreviations and typos.
_ABBREVS = {
    r"all[|°.\s]*o\.?": "allegro",   # All|o, Allo.
    r"all°\.?": "allegro",            # All°, All°.
    r"\ballgero\b": "allegro",        # KernScores typo (missing 'e')
    r"\ballego\b": "allegro",         # KernScores typo (missing 'r')
    r"\ball\.\s": "allegro ",         # "All. molto" → "Allegro molto"
}
_ABBREV_PATS = [(re.compile(k, re.IGNORECASE), v) for k, v in _ABBREVS.items()]


def _expand_abbreviations(text: str) -> str:
    for pat, repl in _ABBREV_PATS:
        text = pat.sub(repl, text)
    return text


def _parse_single_segment(segment: str) -> Optional[float]:
    """Try to resolve one text segment to quarter-note BPM.

    Priority:
    1. Explicit metric: [unit] = number (or M.M. [unit] = number)
    2. Full compound lookup in CLASSIC_TEMPOS (preserves assai/molto/poco)
    3. Longest prefix match
    4. partitura parse_direction classification + base-term fallback
    """
    segment = segment.strip().rstrip(".")
    if not segment:
        return None

    if segment.startswith("tmp") and "musicxml" in segment:
        return None

    # Try explicit metric.
    for pat in (_RE_MM, _RE_METRIC):
        m = pat.search(segment)
        if m:
            unit_str = (m.group(1) or "quarter").strip().lower()
            number = float(m.group(2))
            multiplier = _BEAT_UNITS.get(unit_str)
            if multiplier is not None and number > 0:
                return round(number * multiplier, 2)

    # Expand abbreviations (All°. → Allegro, etc.)
    expanded = _expand_abbreviations(segment)

    # "Tempo di X" / "In tempo di X" → look up X.
    stripped = re.sub(
        r"^(?:in\s+)?tempo\s+d[ie]\s+", "", expanded, flags=re.IGNORECASE
    ).strip()
    if stripped and stripped.lower() != expanded.lower():
        result = _parse_single_segment(stripped)
        if result is not None:
            return result

    # Normalize for compound lookup (strip non-alpha, join).
    normalized = re.sub(r"[^a-z]", "", expanded.lower())
    if not normalized:
        return None

    # Exact match in CLASSIC_TEMPOS (catches compound terms like
    # "allegroassai", "moltovivace", "pocoallegro").
    if normalized in CLASSIC_TEMPOS:
        return float(CLASSIC_TEMPOS[normalized])

    # Longest prefix match (e.g. "allegromoltoconbrio" → "allegro").
    for length in range(len(normalized), 3, -1):
        prefix = normalized[:length]
        if prefix in CLASSIC_TEMPOS:
            return float(CLASSIC_TEMPOS[prefix])

    # Use partitura to classify: only accept ConstantTempoDirection.
    try:
        from partitura.directions import parse_direction
        from partitura.score import ConstantTempoDirection

        dirs = parse_direction(expanded)
        for d in dirs:
            if isinstance(d, ConstantTempoDirection):
                base = getattr(d, "text", None)
                if base and base in CLASSIC_TEMPOS:
                    return float(CLASSIC_TEMPOS[base])
    except Exception:
        pass

    return None


def parse_omd_tempo(omd: str) -> Optional[float]:
    """Parse an !!!OMD value and return quarter-note BPM, or None.

    Priority:
    1. Explicit metric: [unit] = number (or M.M. [unit] = number)
    2. Full compound lookup in CLASSIC_TEMPOS (preserves assai/molto/poco)
    3. partitura parse_direction classification + base-term fallback

    Multi-part OMDs (newline-separated or colon/period-prefixed) are
    split into segments and each is tried independently.
    """
    # Try full string first.
    result = _parse_single_segment(omd)
    if result is not None:
        return result

    # Multi-line: try each line. Kern OMD uses literal "\n" (two chars).
    for sep in ("\n", "\\n"):
        if sep in omd:
            for line in omd.split(sep):
                result = _parse_single_segment(line)
                if result is not None:
                    return result

    # Colon prefix: "Rondo: Allegro" → try after colon.
    for sep in (":", "."):
        if sep in omd:
            after = omd.split(sep, 1)[1]
            result = _parse_single_segment(after)
            if result is not None:
                return result

    return None
