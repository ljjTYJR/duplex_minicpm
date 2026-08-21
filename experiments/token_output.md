- 如果不sft模型，只是单纯手动地修改模型token的排布:

其中 `stub=<think>\n\n</think>\n\n`
此时的token排布是
```
<unit> [img][aud] <|listen|> <think>\n\n</think>\n\n [<tool_call>…</tool_call>] <|act_end|> </unit>
```

```
[push] t=0 listen+stub -> 'I need to ask the user for more information to determine if they want to know about a specific topic or if the'
[push] t=1 listen+stub -> '<tool_call>\n{"name": "get_temperature", "arguments": {"location": "Nairobi"}}\n</tool_call><|im_end|>'
[push] t=2 listen+stub -> '<tool_call>\n{"name": "get_temperature", "arguments": {"location": "Nanjing"}}\n</tool_call><|im_end|>'
[push] t=3 listen+stub -> '<tool_call>\n{"name": "get_temperature", "arguments": {"city": "Nanjing"}}\n</tool_call><|im_end|>'   <-- 问题刚听完
[push] t=4 listen+stub -> '<tool_call>\n{"name": "get_temperature", "arguments": {"city": "Nanjing"}}\n</tool_call><|im_end|>'
[push] t=5 listen+stub -> '<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "Nanjing"}}\n</tool_call><|im_end|>'
[push] t=3 listen only  -> " I don't have access to real-time data, so I can't provide the current temperature in Nanjing.<|turn_eos|>"
```
这说明，添加了`stub`之后，模型自动切换到了 "turn-based" 的chat方式，会自动输出`toll_call`和`<|im_end|>`.
> 说明模型只是学会了在特定token模板下的排布

- 将 audio 静默（输入silence 的音频），然后只通过 text 文本进行查询。

SFT 之后（100 条数据，问题**全部**是 TTS 语音，训练里没有任何文本提问）。三种注入方式：

```
A  <unit> [img][aud] What's the temperature in Nanjing right now? | <|listen|> </unit>   问题挂在媒体 unit 上
B  <unit> What's the temperature in Nanjing right now?            | <|listen|> </unit>   问题独占纯文本 unit
C  <unit> [img][aud] User asks: What's the temperature in ...      | <|speak|>  </unit>   加 "User asks: " 前缀
```

```
[A] t=3 收到问题 -> listen
[A] t=4         -> listen
[A] t=5         -> '<tool_call>{"name":"query_weather","arguments":{"city":"Nanjing"}}</tool_call>'
[A] 回注 39 度后 -> "It's currently 39 degrees Celsius in Nanjing. That's quite hot, be sure to stay hydrated!"

[B] 同 A，调用成功，延迟同样是 2 拍

[C] t=3 -> <|speak|> <|chunk_eos|>          (开口却一个字没说)
[C] t=4 -> ' I will call the'
[C] t=5 -> ' function to query the'
[C] t=6 -> ' weather in Nanjing.'           <-- 只是"说"要调用，没有产出 <tool_call>
```

> A/B 说明 SFT 学到的是跨模态的语义触发，纯语音数据训出的行为迁移到了文本通道；挂在媒体 unit 上还是独占 unit 没有区别。
> C 说明这个能力很脆：多 4 个 token 的旁白式包装（`User asks: `）就退回微调前的行为——与零样本时 duplex/push/text 的表现一致。训练数据里问题只有语音一种形态，上下文从未出现过转述语气。
> 另：文本提问比语音提问晚 2 拍才动手。语音时问题播完当拍即调用，文本时问题第 3 拍就完整到达却等到第 5 拍。推测是训练分布里问题总横跨数秒音频留下的节奏。

注：A 这种"文本与音视频同 unit"的排布需要先修 `streaming_prefill` 的 `pending_logits`（原代码在混合模态下不从文本位置取 logits，decision token 看不到刚注入的文本）。