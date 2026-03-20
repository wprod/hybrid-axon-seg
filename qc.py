"""qc.py — QC filtering with per-fiber rejection reason tracking.

Each rejected fiber gets a short code indicating the *first* failing filter:
  G   = g-ratio out of [MIN_GRATIO, MAX_GRATIO]
  Ø   = axon diameter < MIN_AXON_DIAM_UM
  sol = solidity < MIN_SOLIDITY
  ecc = eccentricity > MAX_AXON_ECCEN
  off = centroid offset > MAX_CENTROID_OFFSET
  brd = touches image border (when EXCLUDE_BORDER=True)

These codes are rendered directly on the overlay image so the clinician can
identify the cause of rejection at a glance.
"""

import pandas as pd

import config

_REASON_LABEL: dict[str, str] = {
    "gratio": "G",
    "axon_diam": "Ø",
    "solidity": "sol",
    "eccen": "ecc",
    "offset": "off",
    "border": "brd",
    "large_lowG": "lgG",  # large fiber + g-ratio < LARGE_FIBER_MIN_GRATIO
}


def apply_qc(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into (passed, rejected) DataFrames.

    The returned *rejected* DataFrame has an extra ``reject_reason`` column
    containing the short code of the first failing filter.
    """
    filters: dict[str, pd.Series] = {}

    if "gratio" in df.columns:
        filters["gratio"] = ~(
            df["gratio"].notna() & df["gratio"].between(config.MIN_GRATIO, config.MAX_GRATIO)
        )
    filters["axon_diam"] = df["axon_diam"] < config.MIN_AXON_DIAM_UM
    if "solidity" in df.columns:
        filters["solidity"] = df["solidity"] < config.MIN_SOLIDITY
    if "eccentricity" in df.columns:
        filters["eccen"] = df["eccentricity"] > config.MAX_AXON_ECCEN
    if "centroid_offset" in df.columns:
        filters["offset"] = df["centroid_offset"] > config.MAX_CENTROID_OFFSET
    if config.EXCLUDE_BORDER and "image_border_touching" in df.columns:
        filters["border"] = df["image_border_touching"].fillna(False).astype(bool)
    if "fiber_diam" in df.columns and "gratio" in df.columns:
        size_thresh = df["fiber_diam"].quantile(config.LARGE_FIBER_PERCENTILE / 100)
        filters["large_lowG"] = (df["fiber_diam"] >= size_thresh) & (
            df["gratio"] < config.LARGE_FIBER_MIN_GRATIO
        )

    reject = pd.Series(False, index=df.index)
    reason = pd.Series("", index=df.index, dtype=str)

    for name, mask in filters.items():
        n = int(mask.sum())
        if n:
            print(f"         QC [{name}]: {n} rejected")
        reason.loc[(reason == "") & mask] = _REASON_LABEL.get(name, name[:3])
        reject |= mask

    df_rej = df[reject].copy()
    df_rej["reject_reason"] = reason[reject].values
    return df[~reject].copy(), df_rej
