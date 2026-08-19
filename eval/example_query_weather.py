"""One fake tool (query_weather) across 2x2x2 cells: {turn-based chat, duplex streaming} x
{question as text, question spoken via TTS on the audio track} x {plain system prompt, one that
orders the model to call the tool}. Same schema, same frames, text out only.
Every cell runs in its own process — one shared model would leak state between them.
Run from repo root: .venv/bin/python eval/example_query_weather.py [seconds] [cell]
"""
import json, math, os, re, subprocess, sys, tempfile

import librosa, numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minicpmo import load_model

CLIP = ("/dataset_rc_rl/suns7@xiaopeng.com/datasets/ego_4d/offline_qa/goalstep"
        "/clips/994616c8-63b0-47bd-a526-0680f4acff9e_557.37_574.18.mp4")
Q = "What's the temperature in Nanjing right now?"
SCHEMA = [{"type": "function", "function": {
    "name": "query_weather", "description": "Get the current temperature of a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
SR, TAIL, WAV = 16000, 8, ".cache/query_weather.wav"
SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 4  # pre-roll length = frame count (1 fps)
CELLS = [f"{m}/{p}/{a}" for p in ("base", "push") for a in ("text", "audio")
         for m in ("turn-based", "duplex")]
CELL = sys.argv[2] if len(sys.argv) > 2 else None
KOKORO_DIR = "/dataset_rc_rl/suns7@xiaopeng.com/models/Kokoro-82M-ONNX"  # ONNX release from github.com/thewh1teagle/kokoro-onnx
KOKORO = ("import sys, soundfile as sf; from kokoro_onnx import Kokoro;"
          "a, sr = Kokoro(sys.argv[2], sys.argv[3]).create(sys.argv[1], voice='af_heart', lang='en-us');"
          "sf.write(sys.argv[4], a, sr)")


def query_weather(city):  # the fake tool
    return {"city": city, "temp_c": 22}


def parse(text):
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    return json.loads(m.group(1)) if m else None


def execute(call):
    """Zero-shot the model often invents its own name/arg, so match loosely and report it."""
    args = call.get("arguments", {})
    matched = call.get("name") == "query_weather" and set(args) == {"city"}
    result = query_weather(args.get("city") or next(iter(args.values()), ""))
    return matched, f"<tool_response>{json.dumps(result)}</tool_response>"


def tts(text, path):
    """Kokoro runs in its own venv — its deps would shadow the training stack. Weights are the
    ONNX release (the HF hub is unreachable here; github.com is)."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(["/root/.venv-kokoro/bin/python", "-c", KOKORO, text,
                        f"{KOKORO_DIR}/kokoro-v1.0.onnx", f"{KOKORO_DIR}/voices-v1.0.bin", path], check=True)
    return librosa.load(path, sr=SR, mono=True)[0]


if CELL is None:  # driver: fan the cells out, one fresh process each
    for cell in CELLS:
        subprocess.run([sys.executable, os.path.abspath(__file__), str(SECONDS), cell])
    sys.exit(0)

tmp = tempfile.mkdtemp()
subprocess.run(["ffmpeg", "-v", "error", "-i", CLIP, "-vf", "fps=1", "-frames:v", str(SECONDS + TAIL),
                f"{tmp}/%02d.jpg"], check=True)
frames = [Image.open(f"{tmp}/{f}").convert("RGB") for f in sorted(os.listdir(tmp))]
question_wav = tts(Q, WAV)
silence = np.zeros(SR, dtype=np.float32)
SECONDS = max(SECONDS, math.ceil(len(question_wav) / SR))  # the spoken question must fit the pre-roll

model, tok = load_model()
PROMPTS = {
    "base": "You are a helpful assistant.",
    "push": ("You are a helpful assistant with access to tools. You cannot know live facts such as "
             "weather or temperature on your own. Whenever the user asks for one, you MUST answer "
             "with a <tool_call> using exactly the function name and argument names given below. "
             "Never reply that you lack access to real-time data."),
}


def tools(prompt):
    """apply_chat_template puts this system text first and the tool block right after it."""
    return (tok.apply_chat_template([{"role": "system", "content": PROMPTS[prompt]}],
                                    tools=SCHEMA, tokenize=False)
            .removeprefix("<|im_start|>system\n").split("<|im_end|>")[0])


def turn_based(prompt, ask):
    """ask='text': question as a string. ask='audio': the spoken question replaces it."""
    heard = question_wav if ask == "audio" else silence
    content = [*frames[:SECONDS], heard] + ([] if ask == "audio" else [Q])
    msgs = [{"role": "system", "content": [tools(prompt)]}, {"role": "user", "content": content}]
    out = model.chat(msgs=msgs, tokenizer=tok, do_sample=False, max_new_tokens=96)
    call = parse(out)
    if not call:
        return {"called": False, "output": out}
    matched, response = execute(call)
    msgs += [{"role": "assistant", "content": [out]}, {"role": "user", "content": [response]}]
    return {"called": True, "call": call, "schema_match": matched,
            "answer": model.chat(msgs=msgs, tokenizer=tok, do_sample=False, max_new_tokens=96)}


def duplex(prompt, ask):
    """ask='text': inject the question as tokens. ask='audio': paste the wav on the track,
    end-anchored at second SECONDS, so the model hears it like any other sound."""
    track = np.zeros((SECONDS + 2 * TAIL) * SR, dtype=np.float32)
    if ask == "audio":
        track[SECONDS * SR - len(question_wav):SECONDS * SR] = question_wav
    s = model.as_duplex(generate_audio=False)
    s.prepare(prefix_system_prompt=tools(prompt))
    chunks = []

    def tick(i):
        s.streaming_prefill(audio_waveform=track[i * SR:(i + 1) * SR], frame_list=[frames[i % len(frames)]])
        r = s.streaming_generate(max_new_speak_tokens_per_chunk=48, decode_mode="greedy")
        chunks.append("listen" if r["is_listen"] else r["text"])
        return r

    def inject(text):
        """Text rides its own unit: prefill opens <unit>, generate closes it with a decision
        token + </unit>. Skipping the generate would leave the unit open and nest the next one."""
        s.streaming_prefill(text_list=[text])
        r = s.streaming_generate(max_new_speak_tokens_per_chunk=48, decode_mode="greedy")
        chunks.append("listen" if r["is_listen"] else r["text"])

    def run_until_eot(start):
        for i in range(start, start + TAIL):
            if tick(i)["end_of_turn"]:
                break
        return parse("".join(c for c in chunks if c != "listen"))

    for i in range(SECONDS):
        tick(i)
    if ask == "text":
        inject(f"User asks: {Q}")
    call = run_until_eot(SECONDS)
    if not call:
        return {"called": False, "chunks": chunks}
    matched, response = execute(call)
    inject(response)
    run_until_eot(SECONDS + TAIL)
    return {"called": True, "call": call, "schema_match": matched, "chunks": chunks}


mode, prompt, ask = CELL.split("/")
result = (turn_based if mode == "turn-based" else duplex)(prompt, ask)
print(f"== {CELL} == seconds={SECONDS} wav={len(question_wav) / SR:.1f}s",
      json.dumps(result, ensure_ascii=False, indent=2))
