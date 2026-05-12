"""
Convert biomedical CURIEs (CHEBI, HGNC, ...) in a ChemProt-style JSON
into human-readable names using Gilda.

Usage:
    pip install gilda
    python id_to_name.py input.json [output.json]

Input format (list of objects):
    {
      "path": ["CHEBI:CHEBI:50113", "HGNC:13731", "raw:estromustine", ...],
      "relations": [["CPR:3", "chemprot"], ...],
      "inferred": "CHEBI:CHEBI:50113 -> HGNC:13731"
    }

Output: same structure, but each id is replaced with
    "<name> (<original_id>)"
when a name is available.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

import gilda

# Matches 'PREFIX:value' OR 'PREFIX:PREFIX:value' (collapses double prefix).
# Examples that match:
#   'CHEBI:CHEBI:50113' -> ('CHEBI', '50113')
#   'CHEBI:50113'       -> ('CHEBI', '50113')
#   'HGNC:644'          -> ('HGNC', '644')
#   'GO:0006915'        -> ('GO',  '0006915')
_CURIE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*):(?:[A-Za-z][A-Za-z0-9]*:)?(.+)$")


# ChemProt relation codes (from the BioCreative VI ChemProt task).
# Gilda does not ground these, so we hard-code them.
CPR_LABELS = {
    "CPR:1": "PART_OF",
    "CPR:2": "REGULATOR",
    "CPR:3": "UPREGULATOR / ACTIVATOR",
    "CPR:4": "DOWNREGULATOR / INHIBITOR",
    "CPR:5": "AGONIST",
    "CPR:6": "ANTAGONIST",
    "CPR:7": "MODULATOR",
    "CPR:8": "COFACTOR",
    "CPR:9": "SUBSTRATE / PRODUCT_OF",
    "CPR:10": "NOT",
}


_name_cache: dict[str, str] = {}


def parse_curie(curie: str) -> tuple[str | None, str | None]:
    """Split a CURIE into (prefix, local_id).

    Handles double-prefix CURIEs ('CHEBI:CHEBI:50113') and skips raw
    text entries. Returns (None, None) when we have no id to look up.
    """
    if curie.startswith("raw:"):
        return None, None
    m = _CURIE_RE.match(curie)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def id_to_name(curie: str) -> str:
    """Return a display label for a CURIE.

    'CHEBI:CHEBI:50113'  -> 'kinase inhibitor (CHEBI:50113)'
    'HGNC:13731'         -> 'AKT1 (HGNC:13731)'
    'raw:estromustine'   -> 'estromustine'
    'CPR:3'              -> 'UPREGULATOR / ACTIVATOR (CPR:3)'
    """
    if curie in _name_cache:
        return _name_cache[curie]

    label: str

    if curie in CPR_LABELS:
        label = f"{CPR_LABELS[curie]} ({curie})"
    elif curie.startswith("raw:"):
        label = curie[len("raw:") :]
    else:
        prefix, ident = parse_curie(curie)
        if prefix is None or ident is None:
            label = curie
        else:
            name = _gilda_lookup(prefix, ident)
            if name:
                # Show a canonical CURIE without the double-prefix quirk.
                clean_curie = f"{prefix}:{ident}"
                label = f"{name} ({clean_curie})"
            else:
                label = curie  # fallback: leave as-is

    _name_cache[curie] = label
    return label


def _gilda_lookup(prefix: str, ident: str) -> str | None:
    """Ask Gilda for the canonical name of a (prefix, id) pair.

    Per Gilda's source (gilda/api.py), the convention is lowercase db
    and bare numeric id: get_names('chebi', '50113'). But to be safe
    against any quirks we also try uppercase db and a prefixed id form,
    and we try 'status=name' (canonical) before falling back to any
    synonym.
    """
    bare = ident
    prefixed = ident if ident.startswith(f"{prefix}:") else f"{prefix}:{ident}"

    db_candidates = [prefix.lower(), prefix.upper(), prefix]
    id_candidates = [bare, prefixed]

    # Prefer canonical name (status='name') before falling back to synonyms.
    for status in ("name", None):
        for db in db_candidates:
            for cand_id in id_candidates:
                try:
                    kwargs = {"status": status} if status else {}
                    names = gilda.get_names(db, cand_id, **kwargs)
                except Exception:
                    names = []
                if names:
                    return names[0]
    return None


def convert_inferred(s: str) -> str:
    """Convert 'A -> B' (or 'A → B') with each side replaced by id_to_name."""
    for arrow in ("→", "->"):
        if arrow in s:
            left, right = [part.strip() for part in s.split(arrow, 1)]
            return f"{id_to_name(left)} {arrow} {id_to_name(right)}"
    return s


def convert(obj: Any) -> Any:
    """Walk the JSON tree and rewrite known id fields."""
    if isinstance(obj, list):
        return [convert(x) for x in obj]
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key == "path" and isinstance(value, list):
                out[key] = [id_to_name(item) for item in value]
            elif key == "relations" and isinstance(value, list):
                out[key] = [
                    [id_to_name(rel) if isinstance(rel, str) and rel.startswith("CPR:") else rel
                     for rel in pair]
                    if isinstance(pair, list) else pair
                    for pair in value
                ]
            elif key == "inferred" and isinstance(value, str):
                out[key] = convert_inferred(value)
            else:
                out[key] = convert(value)
        return out
    return obj


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)

    in_path = argv[1]
    out_path = argv[2] if len(argv) > 2 else in_path.rsplit(".", 1)[0] + ".named.json"

    with open(in_path, encoding="utf-8") as fh:
        data = json.load(fh)

    converted = convert(data)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(converted, fh, indent=2, ensure_ascii=False)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv)
