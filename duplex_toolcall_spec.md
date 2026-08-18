# Duplex tool-call token layout (v1 — direct calls only, planning deferred)

Target: MiniCPM-o 4.5 duplex, text-only output, video+audio input.
Design: DuplexSLA-style unconditional action slot, MiniCPM-style listen gate kept.

## Token inventory

| token | source | status |
|---|---|---|
| `<unit>` `</unit>` `<image>` `</image>` | scaffold feeds | pretrained |
| `<|listen|>` `<|speak|>` `<|chunk_eos|>` `<|turn_eos|>` | model emits | pretrained |
| `<tool_call>` `</tool_call>` | model emits | pretrained (Qwen3 heritage, trained embeddings) |
| `<tool_response>` `</tool_response>` | scaffold injects | pretrained |
| `<|act_end|>` | model emits | **NEW — the only added token** (resize_token_embeddings) |

## Unit grammar (per 1s chunk)

```
<unit> [img_embed_64] [audio_embed_10]
       ( <|listen|>
       | <|speak|> text… (<|chunk_eos|> | <|turn_eos|>) )
       [ <tool_call>{"name":…,"arguments":{…}}</tool_call> ]
       <|act_end|>
</unit>
```

The action slot exists in EVERY unit. Empty slot = bare `<|act_end|>` — an explicitly
supervised noop, which is what solves the class-imbalance problem structurally.

The four states:

```
listen, no action : <unit>[P] <|listen|> <|act_end|> </unit>                         (~95%+)
listen + call     : <unit>[P] <|listen|> <tool_call>{…}</tool_call> <|act_end|> </unit>   (silent call)
speak,  no action : <unit>[P] <|speak|> t… <|chunk_eos|> <|act_end|> </unit>
speak  + call     : <unit>[P] <|speak|> t… <|chunk_eos|> <tool_call>{…}</tool_call> <|act_end|> </unit>
```

## Rules

1. **One call max per unit** (rate limit = 1 Hz action channel).
2. **v1: a call must fit its slot** (act budget ~48 tokens; enforce in the data builder,
   use compact JSON). Cross-unit spillover (DuplexSLA FIFO) deferred with planning.
3. `<tool_call>` JSON format = Qwen chat-template convention (leverage pretrained format).
4. Tool schema: rendered by `tokenizer.apply_chat_template(tools=…)`, passed to
   `prepare(prefix_system_prompt=…)`. Protected from sliding-window eviction.
5. **Result injection** (scaffold, async, at arrival second, lands after audio embeds
   in prefill order):
   `<unit>[img][aud] <tool_response>{…}</tool_response> <decision>… </unit>`
6. While a call is pending: normal listen/speak units; no re-issue (needs negative data).

## Supervision (labels mask)

| span | label |
|---|---|
| system, `<unit>` `</unit>`, perception embeds, `<tool_response>…</tool_response>` | -100 |
| decision token, speak text, `<|chunk_eos|>`/`<|turn_eos|>` | supervised |
| slot content incl. `<tool_call>` json `</tool_call>` and `<|act_end|>` | supervised (noop `<|act_end|>` every unit) |

## Decode-loop change (checkpoint `modeling_minicpmo.py`, duplex `streaming_generate`)

Current loop ends the unit at `<|listen|>` / `<|chunk_eos|>` / `<|turn_eos|>`.
New: after that terminator, keep decoding the action slot until `<|act_end|>`
(own budget, punctuation-forbidden list DISABLED inside the slot — JSON needs `{}":`),
then scaffold feeds `</unit>` as before. Result dict gains `"action": str|None`.

## Deferred (future)

- Planning text in the slot before `<tool_call>` (real-time replacement for `<think>`;
  duplex has no think mechanism). Pure data-format extension, no structural change.
- Cross-unit call spillover (FIFO, `<|act_end|>` deferred, `</unit>` never splits JSON
  mid-escape).
- Multi-call per unit.

## Baselines to beat (zero-shot, measured)

- decision-slot P(`<tool_call>`) with AV stream: 6.2e-07 (rank 16)
- inside forced `<|speak|>` segment with AV: 1.1e-06 (rank 683, refusal basin " I"/" Sorry")
- chat path (patched, normal template): calls zero-shot — upper reference for "how"
