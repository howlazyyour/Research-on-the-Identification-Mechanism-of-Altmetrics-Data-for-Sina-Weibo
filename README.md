# 新浪微博替代计量数据识别机制研究——基于大语言模型相关微博的实证研究

# Research-on-the-Identification-Mechanism-of-Altmetrics-Data-for-Sina-Weibo

本项目展示的为论文《新浪微博替代计量数据识别机制研究——基于大语言模型相关微博的实证》所使用的相关代码以及表格数据。

This project presents the relevant codes and tabular data used in the paper Research on the Identification Mechanism of Altmetrics Data for Sina Weibo: An Empirical Study Based on LLM-Related Weibo.

## 项目结构
```
Research-on-the-Identification-Mechanism-of-Altmetrics-Data-for-Sina-Weibo/
├── 期刊前缀.xlsx # 学术期刊链接前缀库
├── README.md # 项目说明文档
├── data/ # 数据集目录
│ ├── final_weibo_articles.json # Altmetric微博数据集
│ ├── 大模型论文_筛选后数据_链接已替换.csv # 实际为大语言模型主题的微博数据集
│ └── 指向论文的微博_已转换.csv # 大模型关键词搜索得到的微博数据集
├── output/ # 结果输出目录
├── weibo_paper_classifier/ # 源代码目录
│ ├── classifier.py # 核心分类器类
│ ├── data_processors.py # 数据集处理函数
│ ├── evaluator.py # 评估和结果处理函数
│ └── main.py # 主程序入口
```

## 输出文件说明
```
output/altmetric/
├── altmetric_link_classification.csv    # 详细分类结果
├── altmetric_aggregated_results.csv     # 聚合统计结果
└── altmetric_metrics.json               # 性能指标
```

```
output/weibo/
├── weibo_link_classification.csv        # 详细分类结果
├── weibo_aggregated_results.csv         # 聚合统计结果
└── weibo_metrics.json                   # 性能指标
```

## 安装依赖

```bash
pip install pandas numpy matplotlib scikit-learn chardet
```

运行主程序：

```bash
cd weibo_paper_classifier
python main.py
```

根据提示选择要处理的数据集(1/2/3)：

```text
请选择要运行的数据集:
1. Altmetric数据集 (训练/测试)
2. 大模型微博数据集 (验证) # 使用文件“指向论文的微博_已转换.csv”
3. 两者都运行
```

## 数据集说明
1. Altmetric数据集

来源：Altmetric.com官方API

时间范围：2011年10月-2015年6月

规模：10,238条微博记录

用途：框架训练与测试（80%训练集 + 20%测试集）

2. 大模型微博数据集

来源：新浪微博爬取（关键词"大模型"）

时间范围：2009年9月-2024年12月

规模：151,768条微博，4,155条含论文链接， 1673条含大语言模型论文链接

用途：框架泛化能力验证