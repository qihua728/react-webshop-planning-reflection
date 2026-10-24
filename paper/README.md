# 论文源文件

| 文件 | 内容 |
|:--|:--|
| [main.tex](main.tex) | 英文技术报告正文；作者栏留空 |
| [references.bib](references.bib) | 参考文献 |
| [figures/](figures) | 正文引用的图表 |
| [make_figures.py](make_figures.py) | 根据正式分析 CSV 绘制论文图表 |
| [neurips_2025.sty](neurips_2025.sty) | 上游排版模板，保留版权与署名 |

在本目录下使用 LaTeX 与 BibTeX 编译：

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

需要本地 TeX 环境；此发布整理过程未重新编译论文。旧 PDF 未纳入发布目录。
编译后应检查正文、作者栏与 PDF 元数据再发布。

重新绘制论文 PNG 图表还需单独安装 Matplotlib 与 NumPy：

```bash
python -m pip install matplotlib numpy
python make_figures.py
```

统计分析程序直接生成的 SVG 不依赖这两个包。方法、结果和统计细节分别见
[项目首页](../README.md)、[统计报告](../results/formal/analysis/REPORT.md)和
[复现说明](../docs/REPRODUCIBILITY.md)。
