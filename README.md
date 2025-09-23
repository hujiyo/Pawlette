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

## 联系方式

- 作者：hujiyo
- 邮箱：hj1891255909@outlook.com
- 项目状态：[实验性]

---

## 许可证

[MIT]

---
## 进入炼丹交流群

 - 加V：wx17601516389后回复“炼丹交流群”即可

---
