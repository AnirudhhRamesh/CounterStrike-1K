"""Select the exact embedding-window union referenced by retrieval sweep cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from cs2_release.core.io import dataframe_sha256, git_commit, read_parquet, write_json


def select_sweep_windows(
    windows: pd.DataFrame,
    *,
    pairs_roots: list[Path],
) -> tuple[pd.DataFrame, dict]:
    referenced: set[tuple[str, int]] = set()
    contracts = []
    pair_rows = 0
    for root in pairs_roots:
        contract_path = root / "protocol_sweep.json"
        if not contract_path.exists():
            raise FileNotFoundError(contract_path)
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        cells = contract.get("cells")
        if not isinstance(cells, dict) or not cells:
            raise ValueError(f"{contract_path}: missing non-empty cells mapping")
        contracts.append({
            "path": str(contract_path),
            "sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        })
        for cell_name in sorted(cells):
            pair_path = root / f"{cell_name}.parquet"
            if not pair_path.exists():
                raise FileNotFoundError(pair_path)
            pairs = read_parquet(pair_path)
            pair_rows += len(pairs)
            referenced.update(zip(
                pairs["query_eval_window_id"].astype(str),
                pairs["query_pov_idx"].astype(int),
                strict=True,
            ))
            referenced.update(zip(
                pairs["candidate_eval_window_id"].astype(str),
                pairs["candidate_pov_idx"].astype(int),
                strict=True,
            ))

    keys = pd.MultiIndex.from_arrays(
        [
            windows["eval_window_id"].astype(str),
            windows["pov_idx"].astype(int),
        ]
    )
    mask = keys.isin(referenced)
    selected = windows.loc[mask].copy()
    selected_keys = set(zip(
        selected["eval_window_id"].astype(str),
        selected["pov_idx"].astype(int),
        strict=True,
    ))
    missing = referenced - selected_keys
    if missing:
        preview = sorted(missing)[:8]
        raise ValueError(f"{len(missing)} referenced windows are missing; first={preview}")
    selected = selected.sort_values(
        ["map_slug", "match_id", "round_idx", "window_idx", "pov_idx"]
    ).reset_index(drop=True)
    metadata = {
        "schema": "cs2-retrieval-sweep-window-union-v1",
        "status": "pass",
        "source_window_rows": len(windows),
        "selected_window_rows": len(selected),
        "referenced_unique_rows": len(referenced),
        "pair_rows_scanned": pair_rows,
        "pair_contracts": contracts,
        "selected_windows_sha256": dataframe_sha256(selected),
        "git_commit": git_commit(),
    }
    return selected, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--pairs-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    windows = read_parquet(args.windows)
    selected, metadata = select_sweep_windows(
        windows,
        pairs_roots=args.pairs_roots,
    )
    out_path = args.out / "eval_windows.parquet" if args.out.suffix == "" else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(out_path, index=False)
    write_json(out_path.with_suffix(".metadata.json"), metadata)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
