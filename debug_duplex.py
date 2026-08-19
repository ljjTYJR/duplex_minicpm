"""Debugger entry point for duplex mode: one session, one process, no subprocesses, no threads.

    .venv/bin/python debug_duplex.py [seconds]

Or hit F5 on "duplex: step through a session" (.vscode/launch.json).

Breakpoints worth setting:
    minicpmo/modeling_minicpmo.py:2842   streaming_prefill  — feed <unit>, then the perception embeds
    minicpmo/modeling_minicpmo.py:3074   streaming_prefill  — audio feed, the one that fills pending_logits
    minicpmo/modeling_minicpmo.py:3192   streaming_generate — top of the decode loop; j==0 is the decision token
    minicpmo/modeling_minicpmo.py:3215   streaming_generate — listen -> <|tts_bos|> substitution
    minicpmo/modeling_minicpmo.py:3253   streaming_generate — feed </unit>
    minicpmo/utils.py:2144               StreamDecoder.decode — chunk_eos gets its vote on raw logits
    minicpmo/utils.py:2090               StreamDecoder.feed   — the only forward pass in the whole scaffold

Set BREAK_AT_SPEAK=1 to auto-break on the first unit where the model stops listening.
"""
import os, subprocess, sys

import librosa, numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minicpmo import load_model

CLIP = ("/dataset_rc_rl/suns7@xiaopeng.com/datasets/ego_4d/offline_qa/goalstep"
        "/clips/994616c8-63b0-47bd-a526-0680f4acff9e_557.37_574.18.mp4")
WAV = ".cache/query_weather.wav"          # the TTS'd question; silence if absent
FRAMES = ".cache/debug_frames"
SR = 16000
SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ASK_AT = 3                                # second the question starts playing
SCHEMA = ('{"type": "function", "function": {"name": "query_weather", '
          '"description": "Get the current temperature of a city", "parameters": {"type": "object", '
          '"properties": {"city": {"type": "string"}}, "required": ["city"]}}}')
SYSTEM = ("You are a helpful assistant with access to tools.\n\n# Tools\n\n"
          "You may call one or more functions to assist with the user query.\n\n"
          "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
          f"{SCHEMA}\n</tools>\n\nFor each function call, return a json object with function name "
          'and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n'
          '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>')


def media():
    """Frames and waveform are cached — model load is slow enough without redoing ffmpeg."""
    if not os.path.isdir(FRAMES):
        os.makedirs(FRAMES)
        subprocess.run(["ffmpeg", "-v", "error", "-i", CLIP, "-vf", "fps=1", "-frames:v", "64",
                        f"{FRAMES}/%02d.jpg"], check=True)
    frames = [Image.open(f"{FRAMES}/{f}").convert("RGB") for f in sorted(os.listdir(FRAMES))]
    track = np.zeros(SECONDS * SR, dtype=np.float32)
    if os.path.exists(WAV):
        wav = librosa.load(WAV, sr=SR, mono=True)[0][:(SECONDS - ASK_AT) * SR]
        track[ASK_AT * SR:ASK_AT * SR + len(wav)] = wav
    return frames, track


frames, track = media()
model, tok = load_model()                 # step in here for the vendored MiniCPMO/tokenizer wiring
s = model.as_duplex(generate_audio=False)  # text out only; no vocoder, no TTS branch to wade through
s.prepare(prefix_system_prompt=SYSTEM)     # lays <|im_start|>system … <|im_end|> into the KV cache


def emitted(since):
    """Everything the model put out this unit. total_ids only ever grows on the generate side,
    so the tail past `since` is exactly this unit's output — decision token, text, terminator,
    plus the </unit> the scaffold appends (marked, since the model never chose it)."""
    out = []
    for t in s.total_ids[since:]:
        tok_str = tok.convert_ids_to_tokens(t)
        out.append(f"·{tok_str}" if tok_str == "</unit>" else tok_str)
    return " ".join(out)


for i in range(SECONDS):
    # prefill opens <unit>, feeds one frame + one second of audio, parks pending_logits
    s.streaming_prefill(audio_waveform=track[i * SR:(i + 1) * SR], frame_list=[frames[i % len(frames)]])
    if os.environ.get("BREAK_AT_SPEAK") and not s.current_turn_ended:
        breakpoint()                       # the model is mid-answer: next decision token is forced off listen
    mark = len(s.total_ids)
    # generate consumes pending_logits, decodes until a chunk terminator, closes </unit>
    r = s.streaming_generate(max_new_speak_tokens_per_chunk=48, decode_mode="greedy")
    print(f"[{i:02d}] {'listen' if r['is_listen'] else 'speak ':<6} eot={int(r['end_of_turn'])} "
          f"n={r['n_tokens']:<3} {emitted(mark)}")
    if r["text"]:
        print(f"{'':>5} text: {r['text']!r}")

print("\n=== schema: what the LLM actually saw, prefill included, embeds collapsed ===")
print(s.get_session_schema(include_embeddings=True))
