## Pawlette Chat Markup Language (PCML)

### 1. 概述 (Overview)
它的核心职能是充当"Chat Template（对话模板）"的标准化规范，负责将结构化的 OpenAI 格式（List of Messages）无损映射为模型可理解的线性 Token 流（Linear Token Stream），并支持从模型生成流中反向解析出结构化数据。

### 2. 上下文控制流架构 (Contextual Control Schema)
所有以 `[...]` 包裹的标记均为**控制标记 (Control Tokens)**，用于分割不同的消息对象,属于对话流中的**结构性边界**。每个容器对应 OpenAI JSON 中的一个 Message Object。部分容器可以包含元数据。

- **`[SYS]<key>="<value>"[SEP]...[/SYS]`**
  - **Mapping**: `{"role": "system", "content": "..."}`
  - **定义**: 系统指令域。定义模型行为、Persona 及工具定义（Tools Schema）。

- **`[USR]<key>="<value>"[SEP]...[/USR]`**
  - **Mapping**: `{"role": "user", "content": "..."}`

- **`[AST]<key>="<value>"[SEP]...[/AST]`**
  - **Mapping**: `{"role": "assistant", "content": "..."}`
  - **定义**: 模型生成域。包含文本回复、思维链或工具调用请求。

- **`[OBS]<key>="<value>"[SEP]...[/OBS]`**
  - **Mapping**: `{"role": "tool", "tool_call_id": "...", "content": "..."}`
  - **定义**: 观察域 (Observation)。即 OpenAI 中的 Tool Role，用于承载工具调用的返回结果。

### 3. 内容流嵌入 (Content Stream)
本部分定义了容器内部的**内容级**表达。为了与外层结构区分，内部嵌入统一采用 `<...>` XML 风格标记。

- **`<think>...</think>`**
  - **定义**: 显式推理块 (Explicit Reasoning Block)。模型在此区域输出 CoT (Chain of Thought)，在最终映射回 JSON 时，该部分通常被剥离或存入特定字段（如 `reasoning_content`）。

- **`<tools>[{...}{...}...]</tools>`**
  - **定义**: 放在SYS块的末尾，用于定义可用的工具列表,模型更加擅长使用放在此列表内的工具。

- **`<call>{"id":...,"name":...,"arguments":{"k1":"v1",...}}</call>`**
  - **定义**: 结构化动作触发器。模型在需要调用工具时生成此标记。

- **`<end>`**
  - **定义**: 结束符。触发 `stop_token`，标志当前AI Message 对象闭合。

---

### 4. 映射示例 (Bi-directional Mapping Examples)

以下通过一个完整的工具调用对话流程，展示 OpenAI JSON 格式与 PCML Token 流的双向无损转换。

#### 完整对话流程示例

**场景**：用户查询北京天气，模型调用天气工具获取结果后回复。

---

**OpenAI JSON 格式：**
```json
[
  {
    "role": "system",
    "content": "You are a helpful assistant with access to weather tools."
  },
  {
    "role": "user",
    "name": "Alice",
    "content": "What's the weather in Beijing?"
  },
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_abc123",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Beijing\", \"unit\": \"celsius\"}"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_abc123",
    "content": "25°C, Sunny"
  },
  {
    "role": "assistant",
    "content": "The weather in Beijing is 25 degrees Celsius and sunny."
  }
]
```

---

**PCML Token 流：**
```text
[SYS]You are a helpful assistant with access to weather tools.
<tools>[{"name":"get_weather","description":"Get current weather","parameters":{"location":{"type":"string"},"unit":{"type":"string"}}}]</tools>[/SYS]

[USR]name="Alice"[SEP]What's the weather in Beijing?[/USR]

[AST]<think>User wants to know Beijing's weather. I need to call get_weather tool.</think>
<call>{"id": "call_abc123", "name": "get_weather", "arguments": {"location": "Beijing", "unit": "celsius"}}</call><end>[/AST]
             ↓ (系统检测到<call>，执行工具调用)

[OBS]id="call_abc123"[SEP]25°C, Sunny[/OBS]

[AST]The weather in Beijing is 25 degrees Celsius and sunny.<end>[/AST]
```

---

**映射关系详解：**

| OpenAI JSON | PCML Token | 说明 |
|------------|-----------|------|
| `role: "system"` | `[SYS]...[/SYS]` | 系统指令容器 |
| `role: "user"` | `[USR]...[/USR]` | 用户消息容器 |
| `name: "Alice"` | `[USR]name="Alice"[SEP]...[/USR]` | 元数据通过 key-value 形式嵌入 |
| `role: "assistant"` + `tool_calls` | `<call>{...}</call>` | 工具调用使用结构化 JSON |
| CoT / 内部推理 | `<think>...</think>` | 显式推理块（可剥离） |
| `role: "tool"` | `[OBS]id="..."[SEP]...[/OBS]` | 工具返回结果 |
| 消息结束 | `<end>` | 触发 stop_token，系统自动添加闭合标签 |

---

### 5. 技术备忘录 (Technical Memo)

1. **格式鲁棒性**：
   - 解析器（Parser）在处理 `[AST]` 流时，若检测到 `<call>` 标签，应立即挂起文本输出，转为缓冲模式，直到闭合标签出现，将其封装为 `tool_calls` 对象返回。

2. **`<end>` 处理**：
   - `<end>` 由模型生成作为停止信号
   - 系统检测到 `<end>` 后，会自动添加对应的闭合标签（如 `[/AST]`），并移除 `<end>` 本身
   - 对应 OpenAI 格式中的 `finish_reason: "stop"`

3. **元数据传递**：
   - `<key>="<value>"` 形式可嵌入任何控制标记中
   - 常用元数据：`name`（用户名）、`id`（工具调用ID）
