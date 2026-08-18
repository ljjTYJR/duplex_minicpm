# duplex_minicpm

Tool-call action slot for MiniCPM-o 4.5 duplex (text-only out, video+audio in).
Token grammar and v1 rules: duplex_toolcall_spec.md.

## Layout
- minicpmo/  vendored model code — the source of truth for inference/training.
  Load via:  `from minicpmo import load_model`  (weights stay read-only at the
  checkpoint dir; no trust_remote_code, no patches). Divergences from upstream
  are listed in minicpmo/__init__.py.
- data/      jsonl re-render (alert->raise_alert) + sequence builder + consistency check
- train/     LoRA trainer, frozen encoders, cached perception embeddings
- eval/      P(tool_call) probe, alert hit/miss/FA, P(listen) regression
