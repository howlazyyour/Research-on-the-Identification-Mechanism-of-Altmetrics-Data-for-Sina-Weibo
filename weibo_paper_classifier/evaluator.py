# evaluator.py
import json
import os
import pandas as pd
from collections import defaultdict


def aggregate_results(classification_results, id_field='altmetric_id'):
    """按ID聚合结果，取最高可信度"""
    grouped = defaultdict(list)
    for result in classification_results:
        grouped[result[id_field]].append(result)

    confidence_levels = {'high': 4, 'medium': 3, 'low': 2, 'none': 1, 'error': 0}
    aggregated = []

    for item_id, results in grouped.items():
        first_result = results[0]
        any_paper = any(r['is_paper'] for r in results)

        # 计算最高可信度
        max_confidence = 'none'
        max_level = 0
        for r in results:
            if r['is_paper']:
                level = confidence_levels.get(r['confidence'], 0)
                if level > max_level:
                    max_level = level
                    max_confidence = r['confidence']

        aggregated_item = {
            id_field: item_id,
            'any_paper': any_paper,
            'max_confidence': max_confidence,
            'total_links': len(results)
        }

        if 'user_authentication' in first_result:
            aggregated_item.update({
                'user_id': first_result['user_id'],
                'user_authentication': first_result['user_authentication'],
                'retweet_count': first_result['retweet_count'],
                'comment_count': first_result['comment_count'],
                'like_count': first_result['like_count']
            })

        aggregated.append(aggregated_item)

    return aggregated


def calculate_metrics(aggregated_results):
    """计算识别准确率和可信度分布"""
    total = len(aggregated_results)-1
    detected = sum(1 for item in aggregated_results if item['any_paper'])
    accuracy = detected / total if total > 0 else 0

    conf_dist = defaultdict(int)
    for item in aggregated_results:
        conf_dist[item['max_confidence']] += 1

    return {
        'total': total,
        'detected': detected,
        'accuracy': accuracy,
        'confidence_distribution': conf_dist
    }


def save_results(classification_results, aggregated_results, metrics, output_dir="output", dataset_type="altmetric"):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    prefix = dataset_type.lower()

    # 保存详细分类结果
    df_links = pd.DataFrame(classification_results)
    links_file = os.path.join(output_dir, f"{prefix}_link_classification.csv")
    df_links.to_csv(links_file, index=False, encoding='utf-8-sig')
    print(f"[INFO] 详细分类结果已保存到: {links_file}")

    # 保存聚合结果
    df_aggregated = pd.DataFrame(aggregated_results)
    aggregated_file = os.path.join(output_dir, f"{prefix}_aggregated_results.csv")
    df_aggregated.to_csv(aggregated_file, index=False, encoding='utf-8-sig')
    print(f"[INFO] 聚合结果已保存到: {aggregated_file}")

    # 保存指标摘要
    metrics_file = os.path.join(output_dir, f"{prefix}_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 指标摘要已保存到: {metrics_file}")

    # 打印结果
    print(f"\n===== {dataset_type.upper()}数据集评估结果 =====")
    print(f"总样本数: {metrics['total']}")
    print(f"识别为论文的样本数: {metrics['detected']}")
    print(f"识别准确率: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print("可信度分布:")

    conf_order = ['high', 'medium', 'low', 'none']
    for conf in conf_order:
        count = metrics['confidence_distribution'].get(conf, 0)
        percent = count / metrics['total'] if metrics['total'] > 0 else 0
        print(f"  {conf}: {count}条 ({percent * 100:.2f}%)")

    print(f"结果已保存至: {output_dir}")