# classifier.py
import pandas as pd
import os
from datetime import datetime
import urllib.parse
import re
import pickle


class PaperLinkClassifier:
    def __init__(self, excel_path, cache_dir=None):
        """论文链接分类器（四级可信度判定框架）"""
        self.excel_path = os.path.abspath(excel_path)
        self.cache_file = os.path.join(cache_dir or os.path.dirname(self.excel_path),
                                       "journal_patterns_cache.pkl")
        self.last_modified = 0
        self.high_confidence_patterns = []
        self.medium_confidence_patterns = []
        self.journal_map = {}

        # DOI检测
        self.doi_pattern = re.compile(r'\b10\.\d{4,5}/\S+')
        self.doi_indicators = [
            r'doi\.org/', r'doi=', r'doi/', r'doi:', r'dx\.doi\.org/'
        ]

        # 初始化加载模式
        self.reload_patterns()
        # 预编译中可信度学术特征模式
        self._compile_medium_confidence_patterns()

    def _compile_medium_confidence_patterns(self):
        """编译中可信度学术特征模式"""
        # 学术路径关键词
        academic_paths = [
            r'/journals?/', r'/document/',
            r'/abstract/', r'/abs/', r'/fulltext/', r'/paper/',
            r'/preprint/', r'/release/', r'/thesis/', r'/conference/',
            r'/publication/', r'/publications/', r'/pubs/', r'/proceedings/',
            r'/volume/', r'/issue/', r'/fullarticle/', r'/article-abstract/',
            r'/article/abstract/',r'/journals-and-series/', r'/article/view/', '/content/early/'
        ]

        # 学术路径参数
        academic_params = [
            r'\bpii=[^&]+', r'\barnumber=[^&]+',
            r'\bartid=[^&]+', r'\bpaperid=[^&]+'
        ]

        # 编译所有模式
        patterns = []
        for pattern in academic_paths + academic_params:
            patterns.append(re.compile(pattern, re.IGNORECASE))

        self.medium_confidence_patterns = patterns

    def _load_excel_data(self):
        """从Excel文件加载数据"""
        try:
            df = pd.read_excel(self.excel_path)
            if '数据库中论文url的前缀' not in df.columns or 'pdf/epub' not in df.columns:
                raise ValueError("Excel文件缺少必要的列名")
            return df
        except Exception as e:
            print(f"[ERROR] 加载Excel文件出错: {e}")
            return pd.DataFrame()

    def _process_prefix(self, prefix):
        """处理单个前缀"""
        # 处理HTTP/HTTPS
        if prefix.startswith('http://'):
            prefix = 'https?' + prefix[4:]
        elif prefix.startswith('https://'):
            prefix = 'https?' + prefix[5:]

        # 转义特殊字符
        prefix = re.escape(prefix)

        # 处理占位符
        prefix = prefix.replace(r'\?\?\?', '[^/]+')  # ??? → 单一路径段
        prefix = prefix.replace(r'\?\?', '.*?')  # ?? → 任意多段路径

        # 恢复一些特殊字符
        prefix = prefix.replace(r'\:', ':')
        prefix = prefix.replace(r'\.', '.')
        prefix = prefix.replace(r'\?', '?')
        prefix = prefix.replace(r'\(', '(').replace(r'\)', ')')
        prefix = prefix.replace(r'\[', '[').replace(r'\]', ']')

        # 处理特殊字符序列
        prefix = prefix.replace(r'\.\.', '.*?')
        return prefix

    def _generate_patterns_from_df(self, df):
        """从DataFrame生成正则模式"""
        precise_patterns = []
        regex_patterns = []
        journal_map = {}

        for col in ['数据库中论文url的前缀', 'pdf/epub']:
            for idx, value in enumerate(df[col]):
                if pd.isna(value) or value in ['0', 0, '无', 'none', 'null']:
                    continue

                source = df.loc[idx, '来源'] if '来源' in df.columns else f"Column:{col}"
                urls = str(value).replace('；', ';').split(';')

                for url in urls:
                    url = url.strip()
                    if not url or url in ['0', '无', 'none', 'null']:
                        continue

                    processed = self._process_prefix(url)

                    if r'[^/]+' in processed or r'.*?' in processed:
                        try:
                            pattern = re.compile(processed)
                            regex_patterns.append(pattern)
                            journal_map[pattern.pattern] = source
                        except re.error as e:
                            print(f"[WARNING] 无法编译正则表达式: {processed} - {e}")
                    else:
                        try:
                            pattern = re.compile(re.escape(url))
                            precise_patterns.append(pattern)
                            journal_map[pattern.pattern] = source
                        except re.error as e:
                            print(f"[WARNING] 无法编译精确模式: {url} - {e}")

        return precise_patterns, regex_patterns, journal_map

    def _load_cache(self):
        """尝试从缓存加载"""
        if not os.path.exists(self.cache_file):
            return None

        try:
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
                if ('patterns' in data and 'last_modified' in data and
                        'excel_path' in data and data['excel_path'] == self.excel_path):
                    return data
            return None
        except Exception as e:
            print(f"[WARNING] 加载缓存失败: {e}")
            return None

    def _save_cache(self, data):
        """保存数据到缓存"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"[ERROR] 保存缓存失败: {e}")
            return False

    def _check_update_needed(self):
        """检查是否需要更新"""
        if not os.path.exists(self.excel_path):
            print(f"[WARNING] Excel文件不存在: {self.excel_path}")
            return False

        try:
            current_mtime = os.path.getmtime(self.excel_path)
            if current_mtime > self.last_modified or not self.high_confidence_patterns:
                self.last_modified = current_mtime
                return True
            return False
        except Exception as e:
            print(f"[ERROR] 检查文件更新失败: {e}")
            return False

    def reload_patterns(self, force=False):
        """重新加载模式"""
        if not force and not self._check_update_needed():
            return False

        if not force:
            cache_data = self._load_cache()
            if cache_data and cache_data['last_modified'] >= self.last_modified:
                self.high_confidence_patterns = cache_data['patterns']
                self.journal_map = cache_data.get('journal_map', {})
                print(f"[INFO] 从缓存加载期刊模式 (最后修改: {datetime.fromtimestamp(self.last_modified)})")
                return True

        df = self._load_excel_data()
        if df.empty:
            return False

        precise, regex, journal_map = self._generate_patterns_from_df(df)
        self.high_confidence_patterns = precise + regex
        self.journal_map = journal_map

        cache_data = {
            'patterns': self.high_confidence_patterns,
            'last_modified': self.last_modified,
            'excel_path': self.excel_path,
            'journal_map': self.journal_map
        }
        self._save_cache(cache_data)

        print(f"[INFO] 成功加载期刊模式 (模式数: {len(self.high_confidence_patterns)}, "
              f"最后修改: {datetime.fromtimestamp(self.last_modified)})")
        return True

    def contains_doi(self, url):
        """检测URL中是否包含DOI标识"""
        try:
            decoded_url = urllib.parse.unquote(url)

            doi_match = self.doi_pattern.search(decoded_url)
            if doi_match:
                return True, doi_match.group(0).strip()

            for indicator in self.doi_indicators:
                if re.search(indicator, decoded_url, re.IGNORECASE):
                    if "doi=" in url.lower():
                        match = re.search(r'[?&]doi=([^&]+)', url, re.IGNORECASE)
                        if match:
                            return True, urllib.parse.unquote(match.group(1)).strip()
                    elif "doi/" in url.lower():
                        match = re.search(r'doi/([^/&?]+)', url, re.IGNORECASE)
                        if match:
                            return True, urllib.parse.unquote(match.group(1)).strip()
                    elif "doi:" in url.lower():
                        match = re.search(r'doi:([^\s&]+)', url, re.IGNORECASE)
                        if match:
                            return True, urllib.parse.unquote(match.group(1)).strip()
                    return True, "DOI detected"

            return False, None
        except Exception as e:
            print(f"[ERROR] DOI检测失败: {e}")
            return False, None

    def is_high_confidence_paper_link(self, url):
        """检查URL是否匹配高可信度论文模式"""
        try:
            for pattern in self.high_confidence_patterns:
                if pattern.search(url):
                    return True
            return False
        except Exception as e:
            print(f"[ERROR] 高可信度匹配失败: {e}")
            return False

    def get_journal_source(self, url):
        """获取匹配的期刊来源"""
        try:
            for pattern in self.high_confidence_patterns:
                if pattern.search(url):
                    return self.journal_map.get(pattern.pattern, "未知期刊")
            return None
        except Exception as e:
            print(f"[ERROR] 获取期刊来源失败: {e}")
            return None

    def is_medium_confidence_paper_link(self, url):
        """基于学术特征的中可信度检测"""
        try:
            for pattern in self.medium_confidence_patterns:
                if pattern.search(url):
                    return True
            return False
        except Exception as e:
            print(f"[ERROR] 中可信度匹配失败: {e}")
            return False

    def _get_medium_matched_pattern(self, url):
        """获取中可信度匹配的具体模式"""
        for pattern in self.medium_confidence_patterns:
            if pattern.search(url):
                return pattern.pattern
        return None

    def classify(self, url):
        """分级分类论文链接（四级可信度判定框架）"""
        try:
            self.reload_patterns()

            # 1. 高可信度检测（DOI + 前缀库）
            has_doi, doi_value = self.contains_doi(url)
            if has_doi:
                return {"is_paper": True, "confidence": "high", "source": "DOI", "doi": doi_value,
                        "matched_pattern": "DOI"}

            if self.is_high_confidence_paper_link(url):
                source = self.get_journal_source(url) or "期刊"
                return {"is_paper": True, "confidence": "high",
                        "source": source, "doi": None, "matched_pattern": source}

            # 2. 中可信度检测
            if self.is_medium_confidence_paper_link(url):
                matched_pattern = self._get_medium_matched_pattern(url)
                return {"is_paper": True, "confidence": "medium", "source": "学术特征", "doi": None,
                        "matched_pattern": matched_pattern or "学术特征"}

            # 3. 低可信度检测
            low_confidence_keywords = [
                'issn', 'article', 'paper', 'research', 'journal', 'volume', 'issue',
                'conference', 'proceeding', 'abstract', 'citation', 'reference',
                'peer-reviewed', 'scholarly', 'academic', 'preprint', 'archive'
            ]
            for keyword in low_confidence_keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', url, re.IGNORECASE):
                    return {"is_paper": True, "confidence": "low", "source": "弱学术关键词", "doi": None,
                            "matched_pattern": keyword}

            # 4. 未成功识别
            return {"is_paper": False, "confidence": "none", "source": None, "doi": None, "matched_pattern": None}

        except Exception as e:
            print(f"[ERROR] 分类失败: {e}")
            return {"is_paper": False, "confidence": "error", "source": None, "doi": None, "matched_pattern": None}