"""Help dispatcher for `python -m cs2_release`.

Each evaluation has its own runnable module; this file just lists them so
users see the available entry points when they type `python -m cs2_release`.
"""

from __future__ import annotations

ENTRY_POINTS = {
    "pipelines.eval_suite":              "Run the full retrieval + probe pipeline (paper Tables 6-8).",
    "audio_probe.pipeline":              "Run the audio action probe (paper Table 5).",
    "encoders.extract_video":            "Extract frozen video embeddings.",
    "encoders.shard_runner":             "Multi-GPU sharded video embedding extraction.",
    "encoders.extract_audio":            "Extract log-mel audio features.",
    "encoders.merge":                    "Merge sharded embedding outputs.",
    "retrieval.pairs.basic":             "Build cross-POV retrieval pairs.",
    "retrieval.pairs.multipositive":     "Build multi-positive retrieval pairs.",
    "retrieval.pairs.spatial":           "Build state-defined co-located retrieval pairs.",
    "retrieval.eval_basic":              "Evaluate basic cross-POV retrieval.",
    "retrieval.eval_multipositive":      "Evaluate multi-positive / spatial retrieval.",
    "retrieval.eval_pairwise_spatial":   "Pairwise near/far cosine diagnostic.",
    "retrieval.eval_temporal":           "Temporal alignment from same-round/wrong-time predictions.",
    "retrieval.render_qualitative":      "Render qualitative retrieval examples.",
    "action_probe.labels":               "Build per-window action labels from sidecars.",
    "action_probe.train_video":          "Train the frozen-embedding video action probe (paper Table 5).",
    "action_probe.train_multipov":       "Train the k-POV learned action probe.",
    "action_probe.train_offscreen":      "Train the off-screen action imputation probe.",
    "action_probe.coverage":             "1- vs 10-POV action coverage (paper Table 7).",
    "action_probe.off_pov":              "Off-POV miss-rate diagnostic.",
    "audio_probe.train":                 "Train the small audio-only action probe.",
    "global_state.train":                "Train the k-POV global state probe (paper Table 8).",
    "corruption.make_packs":             "Build 10-POV corruption packs.",
    "corruption.train":                  "Train the corruption-detection head.",
    "corruption.eval":                   "Evaluate the corruption head on test packs.",
    "hygiene.eval_split":                "Split-hygiene leakage check.",
    "benchmarks.dataloader_formats":     "Dataloader-format microbenchmark.",
    "core.windows":                      "Build synchronized 1s evaluation windows.",
}

def main() -> int:
    print("CounterStrike-1K release evaluation suite\n")
    print("Run an entry point with: python -m cs2_release.<module> --help\n")
    print("Available modules:\n")
    width = max(len(name) for name in ENTRY_POINTS)
    for name, doc in ENTRY_POINTS.items():
        print(f"  {name.ljust(width)}  {doc}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
