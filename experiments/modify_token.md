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