# Pawlette 爪喵陪伴

> 训练一个二次元少女风格的端侧模型。
> 项目不包含任何预训练权重或数据，只开源模型基础示例。

---

## 项目结构（Minimind风格）

```
.
├── model/           # 模型定义与配置
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

## Pawlette Chat Markup Language (PCML)

完整的 PCML 协议规范请参阅：[PCML.md](./doc/PCML.md)

PCML 是 Pawlette 的对话模板标准化规范，负责将 OpenAI 格式的消息列表无损映射为模型可理解的线性 Token 流，支持思维链、工具调用等高级功能。

## 技术特色

### LongSSM隐藏状态复用机制
Pawlette采用了来自LongSSM论文的技术改进：
- **问题**：传统SSM使用零初始化隐藏状态在长度外推上表现不佳
- **解决方案**：在一个batch内，第一个序列依然使用零初始化，但从第二个序列开始直接复用前一个序列的隐藏状态
- **效果**：显著改善模型的长序列处理能力和长度外推性能，同时节省计算资源

LongSSM论文参考：[LongSSM: On the Length Generalization of State-Space Models](https://arxiv.org/abs/2412.10485)

---

## 历程


- 2026.1.6:**技术改进**:实现LongSSM隐藏状态复用机制,改善长度外推性能 | **方向确立**:借鉴IQuest-coder-v1,Pawlette框架设计打算将方向从"较大参数但稀疏激活"转为"超低参数但重复激活" | 将PSDP协议确立为PCML协议
- 2025.10.1:PSDP协议草稿设计完成
- 2025.9.23:项目结构基本设计完成
- 2025.9.7:Pawlette start ~


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
