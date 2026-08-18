# 计划：MiniCPM duplex SFT 加 action label + tool calling

让 MiniCPM-o 4.5 的 duplex 流在 act 槽输出 action，其余行为（listen/speak 时序、问答）保持不变.
其中 action 槽可以进行 `toll_calling`

## 已就绪（不用再做）

- 训练基座 `MiniCPM-o-4_5-act`（+`<|act_end|>`，行为回归已过）
- 槽语法 `duplex_toolcall_spec.md`；vendored 推理代码 + `load_model`
- 基线数字 `eval/toolcall_baseline.json`（P(tool_call)=5.9e-07 等）与回归脚本 `demo.py`

## Step 1 目标

通过 SfT 的方式让每一个 chunk 均能够显示地输出 `<|act_end|>`.
原本的 tokenizer 中已经有