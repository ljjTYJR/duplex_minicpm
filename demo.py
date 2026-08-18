"""Minimal duplex demo: one stage-1 record, 1s chunks, listen/speak per second.
Run from the repo root: python demo.py
"""
import os, subprocess, sys, tempfile
import librosa, numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minicpmo import load_model

ROOT = "/dataset_rc_rl/suns7@xiaopeng.com/datasets/ego_4d/offline_qa/goalstep"
CLIP = f"{ROOT}/clips/994616c8-63b0-47bd-a526-0680f4acff9e_557.37_574.18.mp4"
WAV = f"{ROOT}/audio/994616c8-63b0-47bd-a526-0680f4acff9e_557.37_574.18.wav"
Q_END, SR, TAIL = 16.0, 16000, 12  # question wav is end-anchored at Q_END (query[0].time)

# media, 1s-aligned: frames at 1 fps; silent track with the question pasted end-anchored
frames_dir = tempfile.mkdtemp()
subprocess.run(["ffmpeg", "-v", "error", "-i", CLIP, "-vf", "fps=1", f"{frames_dir}/%04d.jpg"], check=True)
frames = [Image.open(f"{frames_dir}/{f}").convert("RGB") for f in sorted(os.listdir(frames_dir))]
track = np.zeros(len(frames) * SR, dtype=np.float32)
wav, _ = librosa.load(WAV, sr=SR, mono=True)
track[int(Q_END * SR) - len(wav):int(Q_END * SR)] = wav

model, _ = load_model()
duplex = model.as_duplex(generate_audio=False)
duplex.prepare(prefix_system_prompt="Streaming Omni Conversation.")

print("Q @ 13-16s: What is the person doing in this video?")
for i in range(len(frames) + TAIL):  # TAIL silent seconds so the answer can finish
    audio = track[i * SR:(i + 1) * SR] if i < len(frames) else np.zeros(SR, np.float32)
    duplex.streaming_prefill(audio_waveform=audio, frame_list=[frames[min(i, len(frames) - 1)]])
    r = duplex.streaming_generate(decode_mode="greedy")
    print(f"[{i:2d}]", "listen" if r["is_listen"] else f"speak>{r['text']}", "<eot>" if r["end_of_turn"] else "")
    if r["end_of_turn"] and i >= len(frames):
        break
