# WebShop 消融实验设计

本文档记录四组方法的实验协议。`fixed_0`–`fixed_9` 为开发集，正式测试使用
`fixed_10`–`fixed_109` 共 100 个配对任务。已保存的正式结果位于 `results/formal/`；
后续运行应使用独立输出目录，保留这组结果用于比较。

## 1. 研究问题与假设

- **RQ1：Planning 是否提升约束理解和搜索效率？** 预期提高 reward、成功率，并减少无效搜索。
- **RQ2：Reflection 是否改善失败恢复？** 预期减少非法动作和达到最大步数的任务。
- **RQ3：Planning 与 Reflection 是否互补？** 预期完整方法效果最好，但模型调用和延迟更高。

## 2. 四组方法

| 组别 | 方法标识 | 初始规划 | 过程反思 | WebShop 动作上限 | 模型调用上限 |
|---|---|---:|---:|---:|---:|
| Baseline | `react_baseline` | 0 | 0 | 15 | 15 |
| Planning | `react_planning` | 1 | 0 | 15 | 16 |
| Reflection | `react_reflection` | 0 | 最多2次 | 15 | 17 |
| Full | `react_plan_reflect` | 1 | 最多2次 | 15 | 18 |

四组共享原始 one-shot ReAct 示例和 `think/search/click` 动作空间。Planning 与 Reflection
属于辅助模型调用，不执行网页动作，因此不占用15步环境预算；它们必须计入模型调用、token、
延迟和成本指标，不能隐藏额外开销。

### 2.1 Baseline

保持当前实现不变。每一步由模型直接生成一个 `think[...]`、`search[...]` 或 `click[...]`。
不添加规划、反思、跨任务记忆或策略性重试。

### 2.2 Planning

环境 reset 后、第一次网页动作前增加一次规划调用。模型必须输出：

```text
plan[constraints: ...; search: ...; verify: ...; fallback: ...]
```

- `constraints`：商品类别、属性、规格、价格等硬约束；
- `search`：首选检索词；
- `verify`：购买前必须检查的选项；
- `fallback`：无合适结果时如何改写查询。

规划文本在整个 episode 中保留，并加入后续动作提示词。规划不能直接调用环境，也不能使用
任务外信息。动作生成器及其 one-shot 示例与 Baseline 相同。

### 2.3 Reflection

不做初始规划。当满足下列任一条件时，在下一次网页动作前调用一次反思：

1. 上一步产生非法动作；
2. 完全相同的动作连续出现两次；
3. 执行完第8个环境动作仍未结束，且此前未因该条件反思。

每个 episode 最多反思2次，且两个反思调用不能连续发生。模型必须输出：

```text
reflection[problem: ...; evidence: ...; revised_strategy: ...]
```

反思只读取当前指令和本 episode 最近5个轨迹步骤，不读取其他任务的答案或经验。反思文本加入
后续历史，但不会撤销已经执行的动作。

### 2.4 Full

同时启用 2.2 的一次初始规划和 2.3 的最多两次反思，其他规则不变。Full 不能增加新的触发
条件、环境步数或专属示例，否则无法判断收益是否来自 Planning 与 Reflection 的组合。

## 3. 公平实验条件

四组必须固定以下条件：

- DeepSeek 模型、API 地址、temperature、token 上限和 thinking 设置；
- 同一 WebShop 数据文件、搜索索引、服务进程和任务编号；
- `max_steps=15`、`max_prompt_chars=12000`；
- 相同 one-shot ReAct 示例、环境解析器、动作解析器和网络重试配置；
- 相同成功定义：`reward == 1.0`；
- 相同结果字段和汇总程序。

提示词超长时，只截断最旧的普通轨迹；初始指令、Planning 结果和最近5步必须保留。不得人工修正
任何一组的模型输出。网络或 WebShop 服务错误不算方法失败，应在环境恢复后重跑同一任务并单独
报告环境错误次数。

## 4. 数据划分和运行顺序

### 开发阶段

- 数据：当前 small 环境；
- 任务：`fixed_0`–`fixed_9`；
- 用途：单元测试、提示词格式和触发规则调试；
- 结果：不得进入最终统计。

### 正式阶段

- 任务：`fixed_10`–`fixed_109`，共100个配对任务；
- 四组必须运行完全相同的100个任务；
- 若从 small 切换到完整商品数据，四组必须全部从头重跑，不能混用两种环境的结果。

为减少服务时间和 API 状态变化带来的偏差，按任务交错运行四组，并循环改变组别顺序：

```text
fixed_10: Baseline → Planning → Reflection → Full
fixed_11: Planning → Reflection → Full → Baseline
fixed_12: Reflection → Full → Baseline → Planning
fixed_13: Full → Baseline → Planning → Reflection
```

之后每4个任务重复该顺序。主实验每个“方法×任务”运行一次；若预算允许，可进行3次完整重复，
但必须提前决定，不能只重跑表现不好的方法。

## 5. 日志要求

现有字段全部保留，并为所有方法统一增加：

- `planning_calls`、`reflection_calls`、`total_llm_calls`；
- `input_tokens`、`output_tokens`、`llm_latency_seconds`；
- `plan`；
- `reflections`：每项包含触发条件、发生步数和反思文本；
- `termination_reason`：`purchased`、`max_steps` 或 `environment_error`；
- `prompt_version`、`dataset_mode` 和 WebShop Git commit。

即使某组没有 Planning 或 Reflection，对应字段也要写成 `0`、空字符串或空数组，保证结果可以
直接合并分析。每个正式结果文件中，同一 `task_id` 只能有一个最终有效记录。

## 6. 指标与统计分析

### 主要指标

1. 平均 reward；
2. 严格成功率（`reward == 1.0`）。

### 次要指标

- 平均环境步数；
- 平均模型调用次数及 token 数；
- 非法动作率；
- `max_steps` 比例；
- 平均延迟和估算 API 成本。

所有结果同时报告点估计和95%置信区间。由于四组使用相同任务，应进行配对分析：

- reward 差异：按任务配对 bootstrap 置信区间；
- 严格成功：McNemar 精确检验；
- 步数和调用次数：报告配对中位数差及 bootstrap 区间。

Planning、Reflection、Full 分别与 Baseline 比较，并使用 Holm 方法校正这3次主要比较。报告效应
大小和置信区间，不以 `p < 0.05` 作为唯一结论。

## 7. 结果文件约定

```text
results/formal/react_baseline.jsonl
results/formal/react_baseline_summary.json
results/formal/react_planning.jsonl
results/formal/react_planning_summary.json
results/formal/react_reflection.jsonl
results/formal/react_reflection_summary.json
results/formal/react_plan_reflect.jsonl
results/formal/react_plan_reflect_summary.json
```

保存的统计报告位于 `results/formal/analysis/REPORT.md`，包含效应量、配对检验、
调用开销和局限性；完整技术报告源文件位于 `paper/main.tex`。
