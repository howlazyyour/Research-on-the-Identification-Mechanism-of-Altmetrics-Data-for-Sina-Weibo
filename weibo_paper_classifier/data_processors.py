# data_processors.py
import json
import pandas as pd
import urllib.parse
import chardet
from sklearn.model_selection import train_test_split


def extract_links_from_json(json_file_path):
    """从Altmetric JSON文件提取链接数据"""
    results = []
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for idx, item in enumerate(data):
            citation = item.get('citation', {})
            links = citation.get('links', [])
            first_seen_on = citation.get('first_seen_on')
            altmetric_id = item.get('altmetric_id', f'unknown_{idx}')
            results.append({
                'altmetric_id': altmetric_id,
                'first_seen_on': first_seen_on,
                'links': links,
                'original_data': item
            })
        return results
    except Exception as e:
        print(f"提取链接失败: {str(e)}")
        return []


def split_dataset(link_data, test_size=0.2, random_state=42):
    """将数据集划分为训练集和测试集"""
    print(f"数据集划分：总样本数 {len(link_data)}")
    train_data, test_data = train_test_split(
        link_data,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    print(f"训练集: {len(train_data)}条，测试集: {len(test_data)}条")
    return train_data, test_data


def classify_dataset(classifier, dataset):
    """对Altmetric数据集进行可信度分类"""
    classification_results = []
    for item in dataset:
        altmetric_id = item['altmetric_id']
        first_seen_on = item['first_seen_on']
        for link in item['links']:
            decoded_link = urllib.parse.unquote(link)
            result = classifier.classify(decoded_link)
            classification_results.append({
                'altmetric_id': altmetric_id,
                'first_seen_on': first_seen_on,
                'original_link': link,
                'decoded_link': decoded_link,
                'is_paper': result['is_paper'],
                'confidence': result['confidence'],
                'source': result['source'],
                'doi': result['doi'],
                'matched_pattern': result['matched_pattern']
            })
    return classification_results


def detect_file_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding']


def extract_weibo_links(csv_file_path, link_column='text_links'):
    """从微博CSV文件提取链接数据"""
    results = []
    try:
        encoding = detect_file_encoding(csv_file_path)
        print(f"[INFO] 检测到微博文件编码: {encoding}")

        df = pd.read_csv(csv_file_path, encoding=encoding)
        print(f"[INFO] 成功读取微博数据，共{len(df)}条记录")

        found_column = None
        possible_columns = ['text_links', 'text links', '链接', 'urls', 'links', 'link']

        for col in possible_columns:
            if col in df.columns:
                found_column = col
                break

        if found_column is None:
            for col in df.columns:
                if 'link' in col.lower() or 'url' in col.lower() or '文本链接' in col.lower():
                    found_column = col
                    break

        if found_column is None:
            print(f"[WARNING] 未找到链接列，使用默认列名: {link_column}")
            found_column = link_column

        print(f"[INFO] 使用列 '{found_column}' 作为链接列")

        for idx, row in df.iterrows():
            weibo_id = idx
            links = []

            if pd.notna(row.get(found_column)):
                link_str = str(row[found_column])
                if ',' in link_str:
                    links = [link.strip() for link in link_str.split(',') if link.strip()]
                elif ';' in link_str:
                    links = [link.strip() for link in link_str.split(';') if link.strip()]
                else:
                    if link_str.strip():
                        links = [link_str.strip()]

            results.append({
                'weibo_id': weibo_id,
                'links': links,
                'original_text': str(row.get('text', '')) if 'text' in row else '',
                'user_id': row.get('user_id', ''),
                'created_at': row.get('created_at', ''),
                'retweet_count': row.get('retweet_count', 0),
                'comment_count': row.get('comment_count', 0),
                'like_count': row.get('like_count', 0),
                'user_authentication': row.get('user_authentication', '')
            })

        total_links = sum(len(item['links']) for item in results)
        print(f"[INFO] 共提取到 {len(results)} 条微博，{total_links} 个链接")
        return results

    except Exception as e:
        print(f"[ERROR] 提取微博链接失败: {e}")
        return []


def classify_weibo_dataset(classifier, weibo_data):
    """对微博数据集进行可信度分类"""
    classification_results = []
    total_weibos = len(weibo_data)

    print(f"[INFO] 开始分类微博数据集，共 {total_weibos} 条微博")

    for idx, item in enumerate(weibo_data):
        weibo_id = item['weibo_id']

        for link in item['links']:
            decoded_link = urllib.parse.unquote(link)
            result = classifier.classify(decoded_link)

            classification_results.append({
                'weibo_id': weibo_id,
                'original_text': item['original_text'][:200] + '...' if len(item['original_text']) > 200 else item[
                    'original_text'],
                'user_id': item['user_id'],
                'created_at': item['created_at'],
                'retweet_count': item['retweet_count'],
                'comment_count': item['comment_count'],
                'like_count': item['like_count'],
                'user_authentication': item['user_authentication'],
                'original_link': link,
                'decoded_link': decoded_link,
                'is_paper': result['is_paper'],
                'confidence': result['confidence'],
                'source': result['source'],
                'doi': result['doi'],
                'matched_pattern': result['matched_pattern']
            })

        if (idx + 1) % 500 == 0:
            print(f"[INFO] 已处理 {idx + 1}/{total_weibos} 条微博...")

    print(f"[INFO] 微博数据集分类完成，共处理 {len(classification_results)} 个链接")
    return classification_results
