"""SFT on the duplex overfit set: teach the tool call, keep the rhythm.

Samples are pre-encoded embedding streams (data/make_overfit.py), so the vision and audio encoders
never run here — this is a plain causal-LM fit on inputs_embeds with a mostly -100 label stream.
LoRA on the LLM's attention/MLP only; embed_tokens and lm_head stay frozen because the speak-channel
layout introduces no new token.

    .venv/bin/python train/sft_duplex.py [epochs]
"""
import glob, json, os, re, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minicpmo import load_model

DATA, OUT = ".cache/overfit", ".cache/sft_lora"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
LR, ACCUM, RANK = 1e-4, 4, 16

model, tok = load_model()
llm = model.llm
llm.config.use_cache = False
llm.train()                      # load_model() leaves it in eval(); HF skips checkpointing unless training
llm.gradient_checkpointing_enable()

from peft import LoraConfig, get_peft_model

peft_model = get_peft_model(llm, LoraConfig(
    r=RANK, lora_alpha=2 * RANK, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
peft_model.print_trainable_parameters()

REFUSAL = re.compile(r"don't have|do not have|real-time|cannot|can't provide|I'm sorry|unable to", re.I)


def usable(f):
    """A positive whose answer still refuses after the tool response teaches the model to call the
    tool and then ignore it. Rare (~1 in 50), but exactly the wrong lesson."""
    m = torch.load(f, weights_only=False)["meta"]
    return not (m["positive"] and REFUSAL.search(m["said"]))


allf = sorted(glob.glob(f"{DATA}/*.pt"))
files = [f for f in allf if usable(f)]
print(f"dropped {len(allf) - len(files)} sample(s) whose answer contradicts the tool response")
params = [p for p in llm.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
steps = EPOCHS * -(-len(files) // ACCUM)   # last partial batch still steps
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 1), pct_start=0.1)
print(f"{len(files)} samples, {EPOCHS} epochs, {steps} optimizer steps")

t0 = time.time()
for epoch in range(EPOCHS):
    run, seen = 0.0, 0
    for i, f in enumerate(files):
        d = torch.load(f, weights_only=False)
        # checkpointed blocks need a grad-requiring input; embed_tokens is bypassed here
        e = d["embeds"].to("cuda", torch.bfloat16).unsqueeze(0).requires_grad_(True)
        lab = d["labels"].to("cuda").unsqueeze(0)
        # Only ~3.5% of positions carry a label. Running lm_head over all T would materialise a
        # [T, 151749] fp32 logit tensor (>2.5 GB at T=2100); project just the positions that matter.
        h = llm.model(inputs_embeds=e, use_cache=False).last_hidden_state[0]
        tgt = (lab[0] != -100).nonzero().flatten()
        loss = torch.nn.functional.cross_entropy(llm.lm_head(h[tgt - 1]).float(), lab[0][tgt])
        (loss / ACCUM).backward()
        run += loss.item(); seen += 1
        if (i + 1) % ACCUM == 0 or i + 1 == len(files):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (i + 1) % 20 == 0:
            print(f"  epoch {epoch} [{i+1:3d}/{len(files)}] loss {run/seen:.4f} "
                  f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}G {time.time()-t0:.0f}s")
            run, seen = 0.0, 0
    peft_model.save_pretrained(OUT)
    print(f"epoch {epoch} done -> {OUT}")
print(f"total {time.time()-t0:.0f}s")
