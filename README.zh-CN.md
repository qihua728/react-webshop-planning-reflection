# ReAct on WebShop：规划与反思消融实验

[English](README.md) · [实验报告](results/formal/analysis/REPORT.md) · [代码结构](docs/ARCHITECTURE.md) · [复现说明](docs/REPRODUCIBILITY.md)

在相同的 100 个购物任务上，比较 ReAct、加入规划、加入反思及两者组合。仓库包含四组智能体实现、配置与提示词、400 条正式实验记录，以及配对统计分析。

## 实验结果

每种方法在 `fixed_10`–`fixed_109` 上运行一次。严格成功定义为环境奖励等于 1。

| 方法 | 平均奖励 | 严格成功率 | 平均环境步数 | 平均模型调用 |
|:--|--:|--:|--:|--:|
| ReAct | 0.1360 | 10% | 13.95 | 13.95 |
| + Planning | 0.2755 | 19% | 12.29 | 13.29 |
| + Reflection | 0.1750 | 14% | 13.77 | 15.52 |
| **+ Planning & Reflection** | **0.3658** | **26%** | **11.52** | **13.82** |

![正式实验奖励与成功率](results/formal/analysis/method_performance.svg)

组合方法相对 ReAct 的平均奖励提高 0.230，配对 bootstrap 的 95% 置信区间为 [0.136, 0.327]。成功率增加 **16 个百分点**。它减少了环境步数，但平均耗时从 67.72 秒增加到 115.57 秒。

这是一组固定任务上的结果。规划与反思的交互效应区间跨过零，不能据此断言二者存在统计上的协同效应。Reflection 单独使用时，奖励提升的配对区间也包含零。

## 方法与实现

- 四组共享环境接口、动作循环、模型设置、示例及 15 步预算。
- Planning 在首次动作前调用一次，提取购物约束与搜索计划。
- Reflection 最多调用两次；触发条件为非法动作、相邻两次动作相同，或达到配置中的中点仍未完成。
- 规划和反思计入模型调用、token 与延迟，不占环境动作步数。

主要代码位于 `src/webshop_baseline/`；四组方法位于 `agents/`，共享循环位于 `agents/base.py`。模块对应关系见[结构说明](docs/ARCHITECTURE.md)。

## 先查看结果

不需要 API Key、WebShop 服务或第三方 Python 依赖：

```bash
git clone https://github.com/qihua728/react-webshop-planning-reflection.git
cd react-webshop-planning-reflection
python scripts/verify_results.py
```

校验程序只读检查任务配对、记录完整性及汇总数值。重算置信区间、检验和图表可运行 `python analysis/analyze_formal.py`；该命令耗时更长，并重写 `results/formal/analysis/`。

## 安装与测试

需要 Python 3.10+。在仓库根目录运行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux/macOS 将解释器路径换成 `.venv/bin/python`。整理时现有 18 项单元测试已在本地通过。CI 工作流执行测试和离线数据校验，不调用付费 API；本地验证不代表已完成线上端到端复现。

## 运行新实验

需要独立部署的 WebShop、商品数据、搜索索引和 DeepSeek API Key。**完整环境暂不能仅靠此仓库从零复现**：历史元数据引用的 `full-streaming-ijson-v1` 补丁尚未找回。版本记录和缺失项见[复现说明](docs/REPRODUCIBILITY.md)。

已有服务正常监听 `http://localhost:3000` 时：

```powershell
$env:DEEPSEEK_API_KEY="your-api-key"
.venv\Scripts\python.exe -m webshop_baseline --config configs/baseline_frozen.yaml check-env --task-id fixed_0
.venv\Scripts\python.exe -m webshop_baseline --config configs/baseline_frozen.yaml run --start 0 --count 1
```

`check-env` 不调用模型；`run` 会调用付费 API。开发配置写入独立于正式记录的目录。新正式实验需设置新的输出路径，保留仓库中已有的 400 条记录。

## 文档与来源

- [实验协议](EXPERIMENTS.md)：任务划分、动作预算和统计方法。
- [统计报告](results/formal/analysis/REPORT.md)：完整指标、配对检验与局限。
- [论文源文件](paper/main.tex)：正文和参考文献；旧 PDF 未纳入此版本。
- [ReAct](https://github.com/ysymyth/ReAct) 与 [WebShop](https://github.com/princeton-nlp/WebShop)：框架与环境来源。

上游模板的版权署名保留；项目原创文件暂未选择开源许可证。
