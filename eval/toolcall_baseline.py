"""Zero-shot tool-call baseline on the training-base checkpoint (-act, vocab 151749).

2x2 matrix, same question/schema/frame everywhere: {chat, duplex} x {text, +vision/audio}.
Chat runs the full loop when a call appears; duplex cells probe the decision distribution
(now including <|act_end|>). Results -> eval/toolcall_baseline.json — the numbers SFT must beat.

Expected vs the pre-repo run: A2 now CALLS (vendored code no longer force-switches to the
TTS template on audio input); duplex stays suppressed until fine-tuned.
Run from repo root: python eval/toolcall_baseline.py
"""
import json, os, re, subprocess, sys, tempfile

import numpy as np, torch, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minicpmo import load_model

CLIP = ("/dataset_rc_rl/suns7@xiaopeng.com/datasets/ego_4d/offline_qa/goalstep"
        "/clips/994616c8-63b0-47bd-a526-0680f4acff9e_557.37_574.18.mp4")
Q = "What's the temperature in Nanjing right now?"
SCHEMA = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
PROBE = ["<|listen|>", "<|speak|>", "<tool_call>", "<|act_end|>"]

model, tok = load_model()
TOOLS = (tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}],
                                 tools=SCHEMA, tokenize=False)
         .removeprefix("<|im_start|>system\n").split("<|im_end|>")[0])
jpg = tempfile.mktemp(suffix=".jpg")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "2", "-i", CLIP, "-frames:v", "1", jpg], check=True)
frame, silence = Image.open(jpg).convert("RGB"), np.zeros(16000, dtype=np.float32)
report = {}


def chat_case(name, content):
    msgs = [{"role": "system", "content": [TOOLS]}, {"role": "user", "content": content}]
    out = model.chat(msgs=msgs, tokenizer=tok, do_sample=False, max_new_tokens=96)
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", out, re.S)
    cell = {"tool_called": bool(m), "output": out}
    if m:  # complete the loop: parse -> fake executor -> feed result back
        call, result = json.loads(m.group(1)), {"temp": "22C", "sky": "cloudy"}
        msgs += [{"role": "assistant", "content": [out]},
                 {"role": "user", "content": [f"<tool_response>{json.dumps(result)}</tool_response>"]}]
        cell |= {"parsed_call": call, "tool_result": result,
                 "final_answer": model.chat(msgs=msgs, tokenizer=tok, do_sample=False, max_new_tokens=96)}
    report[name] = cell
    print(f"{name}| called={cell['tool_called']}  {out[:80]!r}")


def duplex_case(name, with_av):
    d = model.as_duplex(generate_audio=False)
    d.prepare(prefix_system_prompt=TOOLS)
    if with_av:
        for _ in range(2):  # two silent seconds of video+audio
            d.streaming_prefill(audio_waveform=silence, frame_list=[frame], max_slice_nums=1)
            d.streaming_generate(max_new_speak_tokens_per_chunk=20, decode_mode="greedy")
    d.streaming_prefill(text_list=[f"User asks: {Q} Please use the tool."])
    p = F.softmax(d.pending_logits.float()[0], -1)
    probs = {t: float(p[tok.convert_tokens_to_ids(t)]) for t in PROBE}
    r = d.streaming_generate(max_new_speak_tokens_per_chunk=48, decode_mode="greedy")
    decision = "listen" if r["is_listen"] else r["text"]
    report[name] = {"tool_called": "<tool_call>" in decision, "decision": decision, "decision_probs": probs}
    print(f"{name}| P(tool)={probs['<tool_call>']:.1e} P(act_end)={probs['<|act_end|>']:.1e} "
          f"P(listen)={probs['<|listen|>']:.2f}  decision={decision!r}")


chat_case("A1_chat_text", [Q])
chat_case("A2_chat_vision_audio", [frame, silence, Q])
duplex_case("B1_duplex_vision_audio", with_av=True)
duplex_case("B2_duplex_text", with_av=False)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toolcall_baseline.json")
json.dump(report, open(out_path, "w"), indent=2, ensure_ascii=False)
print("saved ->", out_path)
