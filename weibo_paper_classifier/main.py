# main.py
from classifier import PaperLinkClassifier
from data_processors import *
from evaluator import *
import os


def process_altmetric_dataset(classifier, json_path, output_dir):
    """处理Altmetric数据集"""
    print(f"\n[INFO] 开始处理Altmetric数据集: {json_path}")

    link_data = extract_links_from_json(json_path)
    if not link_data:
        print("[ERROR] 未提取到Altmetric数据")
        return

    train_data, test_data = split_dataset(link_data, test_size=0.2, random_state=42)

    print("[INFO] 分析测试集可信度...")
    test_classification = classify_dataset(classifier, test_data)
    test_aggregated = aggregate_results(test_classification, id_field='altmetric_id')
    test_metrics = calculate_metrics(test_aggregated)

    save_results(test_classification, test_aggregated, test_metrics,
                 os.path.join(output_dir, "altmetric"), dataset_type="altmetric")


def process_weibo_dataset(classifier, csv_path, output_dir):
    """处理微博数据集"""
    print(f"\n[INFO] 开始处理微博数据集: {csv_path}")

    weibo_data = extract_weibo_links(csv_path)
    if not weibo_data:
        print("[ERROR] 未提取到微博数据")
        return

    print("[INFO] 分析微博数据集可信度...")
    weibo_classification = classify_weibo_dataset(classifier, weibo_data)
    weibo_aggregated = aggregate_results(weibo_classification, id_field='weibo_id')
    weibo_metrics = calculate_metrics(weibo_aggregated)

    save_results(weibo_classification, weibo_aggregated, weibo_metrics,
                 os.path.join(output_dir, "weibo"), dataset_type="weibo")


def main():
    """主函数"""
    # 获取当前文件所在目录的父目录（项目根目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # weibo_paper_classifier的父目录

    print(f"[DEBUG] 当前代码目录: {current_dir}")
    print(f"[DEBUG] 项目根目录: {project_root}")

    # 配置文件路径（使用相对路径）
    CONFIG = {
        'EXCEL_PATH': os.path.join(project_root, "前缀.xlsx"),
        'ALTIMETRIC_JSON_PATH': os.path.join(project_root, "data", "final_weibo_articles.json"),
        'WEIBO_CSV_PATH': os.path.join(project_root, "data", "指向论文的微博_已转换.csv"),
        'OUTPUT_DIR': os.path.join(project_root, "output")
    }

    # 打印路径确认
    print(f"[INFO] Excel路径: {CONFIG['EXCEL_PATH']}")
    print(f"[INFO] Altmetric JSON路径: {CONFIG['ALTIMETRIC_JSON_PATH']}")
    print(f"[INFO] 微博CSV路径: {CONFIG['WEIBO_CSV_PATH']}")
    print(f"[INFO] 输出目录: {CONFIG['OUTPUT_DIR']}")

    # 检查必要的文件是否存在
    for key, path in CONFIG.items():
        if key != 'OUTPUT_DIR':  # OUTPUT_DIR不需要预先存在
            if not os.path.exists(path):
                print(f"[ERROR] 文件不存在: {path}")
                return

    print("=" * 60)
    print("学术链接四级可信度判定框架")
    print("=" * 60)

    # 初始化分类器
    print(f"\n[INFO] 初始化分类器，使用期刊前缀库: {CONFIG['EXCEL_PATH']}")
    classifier = PaperLinkClassifier(CONFIG['EXCEL_PATH'])

    # 选择要运行的数据集
    print("\n请选择要运行的数据集:")
    print("1. Altmetric数据集 (训练/测试)")
    print("2. 大模型微博数据集 (验证)")
    print("3. 两者都运行")

    choice = input("请输入选择 (1/2/3): ").strip()

    if choice in ['1', '3']:
        process_altmetric_dataset(classifier, CONFIG['ALTIMETRIC_JSON_PATH'], CONFIG['OUTPUT_DIR'])

    if choice in ['2', '3']:
        process_weibo_dataset(classifier, CONFIG['WEIBO_CSV_PATH'], CONFIG['OUTPUT_DIR'])

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()