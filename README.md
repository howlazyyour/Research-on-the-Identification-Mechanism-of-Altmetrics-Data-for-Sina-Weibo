# Research-on-the-Identification-Mechanism-of-Altmetrics-Data-for-Sina-Weibo
本项目展示的为论文《新浪微博替代计量数据识别机制研究——基于大语言模型相关微博的实证》所使用的相关代码以及表格数据。

This project presents the relevant codes and tabular data used in the paper Research on the Identification Mechanism of Altmetrics Data for Sina Weibo: An Empirical Study Based on LLM-Related Weibo.

# 新浪微博替代计量数据识别机制研究

基于大语言模型相关微博的实证研究

## 项目结构
Research-on-the-Identification-Mechanism-of-Altmetrics-Data-for-Sina-Weibo/
├── 期刊前缀.xlsx # 学术期刊链接前缀库
├── README.md # 项目说明文档
├── data/ # 数据集目录
│ ├── final_weibo_articles.json # Altmetric微博数据集
│ └── 指向论文的微博_已转换.csv # 大模型微博数据集
├── output/ # 结果输出目录
├── weibo_paper_classifier/ # 源代码目录
│ ├── classifier.py # 核心分类器类
│ ├── data_processors.py # 数据集处理函数
│ ├── evaluator.py # 评估和结果处理函数
│ └── main.py # 主程序入口
└── .gitignore # Git忽略文件

## 安装依赖

```bash
pip install pandas numpy matplotlib scikit-learn chardet

使用说明
确保项目结构正确，数据文件放置在data目录下

运行主程序：

```bash
cd weibo_paper_classifier
python main.py

根据提示选择要处理的数据集