# Pawlette 爪喵陪伴

> 训练一个二次元少女风格的端侧模型。
> 项目不包含任何预训练权重或数据，只开源模型基础示例。

---

## 项目结构（Minimind风格）

```
.
├── model/           # 模型定义与配置
├── trainer/         # 训练脚本
├── dataset/         # 数据加载与预处理
├── out/             # 输出目录（检查点、日志等）
└── environment.yml  # 依赖列表
other files...

```

> 详细文件随着后续更迭会有所改动，但是大致目录基本不会变

---

## 预训练流程

### 1. 环境安装

- 作者对配置环境也很懵QvQ，这里我就直接将相关的环境依赖都保存在environment.yml里可以参考

### 2. 准备数据

- 数据格式：请使用 **JSONL**，字段名为`"text"`，内容。
- 将数据文件放入 `dataset/` 目录下。
- 如需自定义数据读取逻辑，请修改 `dataset/` 中的对应文件。

### 3. 准备分词器

- 分词器文件（`tokenizer.json`, `config.json` 等）应放置于 `model/` 或指定路径。Pawlette暂时使用Minimind项目的分词器进行训练。
- 可使用已有模型（如 `Qwen`、`ChatGLM`、`GPT-2`）的分词器，或自行训练。

### 4. 启动训练

```bash
cd trainer/
python train_pretrain.py
```

> 🔧 所有参数均可自行调整

---

## Pawlette Structured Dialogue Protocol(PSDP)

Pawlette结构化对话协议（设计草案）
协议分为两种维度

### 上下文控制流结构：

默认：
- `[OTHER]user[SEP]...[/OTHER]`:其它用户的对话
- `[AI]...[/AI]`:人工智能的对话

其它功能：
- `[RET]...[/RET]`:tool调用结果回调
- `[SYS]...[/SYS]`:系统工具提示(提供工具使用方法)
- `[IMG]...[/IMG]`:图像流

### 内置输出流特殊结构

默认：
- `... <ed>`:模型输出终止符
- `...<tk>...</tk>...`:模型思考标记符
- `...<lb>...</lb>...`:look back回看历史专用工具，将模型有点忘记内容的结构块（比如图片，文档）插到下文

其它功能通过工具调用：
- `...<to>...</to>...`:工具调用

<details>
<summary>点击展开示例</summary>

```
[SYS]001[SEP]当前可用工具：
1. search_web(query: str) -> str: 网络搜索工具
2. calculate(expression: str) -> float: 数学计算工具
3. get_weather(city: str) -> dict: 获取天气信息[/SYS]
[OTHER]002[SEP]小明[SEP]你好！能帮我查一下北京今天的天气吗？顺便帮我算一下25*36等于多少[/OTHER]
[AI]003[SEP]<tk>用户问了两个问题：1.查询北京天气 2.计算25*36，我需要调用两个工具</tk>好的，让我帮你查询一下！<to>get_weather(city="北京")</to><to>calculate(expression="25*36")</to>[/AI]

[RET]004[SEP]get_weather结果: {"city": "北京", "temp": 15, "condition": "晴"}
calculate结果: 900[/RET]

[AI]005[SEP]<tk>工具返回了结果，现在组织回复</tk>北京今天天气晴朗，温度15度～很适合出门呢！另外25×36=900哦<ed>[/AI]

[OTHER]006[SEP]小红[SEP]我也想知道上海的天气[/OTHER]

[AI]007[SEP]<tk>小红想查上海天气</tk>让我帮你查一下～<to>get_weather(city="上海")</to>[/AI]

[RET]008[SEP]get_weather结果: {"city": "上海", "temp": 18, "condition": "多云"}[/RET]

[AI]009[SEP]上海今天多云，温度18度，比北京稍微暖和一点呢<ed>[/AI]

[IMG]010[SEP]XXXXXXX图片tokenXXXXXXX[/IMG]

[OTHER]011[SEP]小明[SEP]这张图片里是什么？[/OTHER]

[AI]012[SEP]<tk>需要理解图片内容</tk><lb>010</lb>[IMG]010[SEP]XXXXXXX图片tokenXXXXXXX[/IMG]这张图片中我看到了一只可爱的小猫咪，它有着橘色的毛发，正趴在窗台上晒太阳～<ed>[/AI]

[DOC]013[SEP]《人工智能发展报告2025》
第一章：深度学习的最新进展
本章节介绍了Transformer架构的演进...[文档内容省略]...[/DOC]

[OTHER]014[SEP]小红[SEP]刚才那份文档的主要内容是什么？另外帮我搜索一下"量子计算最新进展"[/OTHER]

[AI]015[SEP]<tk>用户问了文档总结和网络搜索两件事</tk><lb>013</lb>[DOC]013[SEP]《人工智能发展报告2025》
第一章：深度学习的最新进展
本章节介绍了Transformer架构的演进...[文档内容省略]...[/DOC]根据刚才的文档，主要讲述了深度学习和Transformer架构的演进～让我再帮你搜索量子计算的信息<to>search_web(query="量子计算最新进展")</to>[/AI]

[RET]016[SEP]search_web结果: 2025年量子计算领域取得重大突破，IBM发布了1000量子比特的量子处理器...[/RET]

[AI]017[SEP]<tk>整合搜索结果回复</tk>搜索结果显示，2025年量子计算领域很火热呢！IBM已经发布了1000量子比特的处理器，这是一个重大突破～<ed>[/AI]

[OTHER]018[SEP]小明[SEP]太棒了！谢谢你[/OTHER]

[AI]019[SEP]<tk>用户表示感谢，友好回应</tk>不客气～很高兴能帮到你们！有什么问题随时问我哦<ed>[/AI]
```

</details>


## 联系方式

- hujiyo
- 邮箱：hj1891255909@outlook.com
- 项目状态：[实验性]

---

## 许可证

[MIT]

---
## 进入炼丹交流群

 - 加V：wx17601516389后回复“炼丹交流群”即可

---
