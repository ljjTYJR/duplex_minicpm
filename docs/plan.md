# 计划：MiniCPM duplex SFT 加 tool calling

让 MiniCPM-o 4.5 的 duplex 在 speak 的时候能够输出工具调用 `tool_calling`，其余行为（listen/speak 时序、问答）保持不变.

## 已就绪（不用再做）

- 训练基座 `MiniCPM-o-4_5-act`（+`<|act_end|>`，行为回归已过）
- 槽语法 `duplex_toolcall_spec.md`；vendored 推理代码 + `load_model`
- 基线数字 `eval/toolcall_baseline.json`（P(tool_call)=5.9e-07 等）与回归脚本 `demo.py`

## Step 1 目标

通过 SfT 的方式让每一个 chunk 均能够显示地输出 `<|act_end|>`.
原本的 tokenizer 中已经有 tool_call 等一些能力.

目标一个 chunk=1s 的 token 排布:

```
<unit> [img_embed_64] [audio_embed_10]
       ( <|listen|>
       | <|speak|> text… (<|chunk_eos|> | <|turn_eos|>) )
       [ <tool_call>{"name":…,"arguments":{…}}</tool_call> ]
</unit>
```
相比较而言，如果是 **turn-based** 的对话方式下，token的排布如下:

```
<|im_start|>user
<image_id>0</image_id><image>[unk*64]</image><slice>[unk*64]</slice><slice>[unk*64]</slice>
<|audio_start|>[unk*22]<|audio_end|><|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "query_weather", "arguments": {"city": "Nanjing"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>{"city": "Nanjing", "temp_c": 22}</tool_response><|im_end|>
<|im_start|>assistant
<think>
</think>
```

其中 <|act_end|> 总是会默认出现的 special token;
[ <tool_call>{"name":…,"arguments":{…}}</tool_call> ] 则是不一定会出现，只有需要调用工具的时候才会出现，调用工具不一定完全通过 <|speak|>
也有可能静默状态下调用 <tool_call>
