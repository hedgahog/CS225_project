from pathlib import Path


def _find_project_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / "Codes_Solhee_Tucker").is_dir():
            return p
        p = p.parent
    raise FileNotFoundError(f"Cannot find CS225_project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).parent)

# --- Grounding -----------------------------------------------------------
SCORE_THRESHOLD = 0.6
# "keep_as_raw": retain triple with null grounding fields
# "drop":        discard triples where either entity fails to ground
UNGROUNDED_POLICY = "keep_as_raw"

# --- Input paths ---------------------------------------------------------
CHEMPROT_INPUT = (
    PROJECT_ROOT
    / "chemprot-relexner-pipeline-main"
    / "chemprot-relexner-pipeline-main"
    / "rl_files"
    / "merged_relation_predictions_all_models.json"
)
EUADR_INPUT = PROJECT_ROOT / "298Code_Tran_Brian" / "euadr_full.json"

# --- Output paths --------------------------------------------------------
OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

NORMALIZED_CHEMPROT = OUTPUTS_DIR / "gilda_normalized_chemprot_triples.json"
NORMALIZED_EUADR    = OUTPUTS_DIR / "gilda_normalized_euadr_triples.json"
NORMALIZED_KG       = OUTPUTS_DIR / "normalized_unified_kg.graphml"
GROUNDING_CACHE     = OUTPUTS_DIR / ".grounding_cache.json"
COMPARISON_REPORT   = OUTPUTS_DIR / "comparison_report.md"
