"""Webスクレイピング機能

玄界システム公式サイトからの文書取得を行う
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Optional, Set, Dict, Any
import time
import logging
from datetime import datetime
import chardet

from ..models.document import Document
from ..utils.logging import get_logger


class WebScraper:
    """
    Webスクレイピングクラス
    
    玄界システム公式サイトから文書を取得・処理するクラス
    """
    
    def __init__(
        self,
        base_url: str = "https://www.cc.kyushu-u.ac.jp/scp/",
        request_delay: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30,
        user_agent: str = "Genkai RAG System 1.0"
    ):
        """
        WebScraperを初期化
        
        Args:
            base_url: スクレイピング対象のベースURL
            request_delay: リクエスト間の遅延時間（秒）
            max_retries: 最大リトライ回数
            timeout: リクエストタイムアウト（秒）
            user_agent: User-Agentヘッダー
        """
        self.base_url = base_url
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.user_agent = user_agent
        
        self.logger = get_logger("scraper")
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # 訪問済みURLを追跡
        self.visited_urls: Set[str] = set()
        
    def scrape_website(self, base_url: Optional[str] = None) -> List[Document]:
        """
        ウェブサイト全体をスクレイピング
        
        Args:
            base_url: スクレイピング対象のベースURL（省略時はインスタンス設定を使用）
            
        Returns:
            取得した文書のリスト
        """
        if base_url is None:
            base_url = self.base_url
            
        self.logger.info(f"ウェブサイトのスクレイピングを開始: {base_url}")
        
        documents = []
        urls_to_visit = [base_url]
        
        while urls_to_visit:
            url = urls_to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
                
            try:
                # ページを取得
                response = self._fetch_page(url)
                if response is None:
                    continue
                
                # 文書を抽出
                document = self._extract_document(response, url)
                if document and document.is_valid():
                    documents.append(document)
                    self.logger.info(f"文書を取得: {document.title} ({url})")
                
                # 同一ドメイン内のリンクを収集
                new_urls = self._extract_links(response, url)
                for new_url in new_urls:
                    if new_url not in self.visited_urls and new_url not in urls_to_visit:
                        urls_to_visit.append(new_url)
                
                self.visited_urls.add(url)
                
                # リクエスト間の遅延
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
                    
            except Exception as e:
                self.logger.error(f"URL {url} の処理中にエラー: {e}")
                continue
        
        self.logger.info(f"スクレイピング完了: {len(documents)}個の文書を取得")
        return documents
    
    def scrape_single_page(self, url: str) -> Optional[Document]:
        """
        単一ページをスクレイピング
        
        Args:
            url: スクレイピング対象のURL
            
        Returns:
            取得した文書（取得失敗時はNone）
        """
        try:
            response = self._fetch_page(url)
            if response is None:
                return None
                
            document = self._extract_document(response, url)
            if document and document.is_valid():
                self.logger.info(f"単一ページを取得: {document.title} ({url})")
                return document
            else:
                self.logger.warning(f"無効な文書: {url}")
                return None
                
        except Exception as e:
            self.logger.error(f"単一ページ取得エラー ({url}): {e}")
            return None
    
    def _fetch_page(self, url: str) -> Optional[requests.Response]:
        """
        ページを取得（リトライ機能付き）
        
        Args:
            url: 取得するURL
            
        Returns:
            レスポンスオブジェクト（取得失敗時はNone）
        """
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"ページ取得試行 {attempt + 1}/{self.max_retries + 1}: {url}")
                
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # 文字エンコーディングを処理
                response = self.handle_japanese_encoding(response)
                
                return response
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"ページ取得失敗 (試行 {attempt + 1}): {url} - {e}")
                
                if attempt < self.max_retries:
                    # 指数バックオフでリトライ
                    wait_time = (2 ** attempt) * self.request_delay
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"ページ取得に失敗: {url}")
                    return None
        
        return None
    
    def handle_japanese_encoding(self, response: requests.Response) -> requests.Response:
        """
        日本語エンコーディングを適切に処理
        
        Args:
            response: HTTPレスポンス
            
        Returns:
            エンコーディング処理済みのレスポンス
        """
        # Content-Typeヘッダーからエンコーディングを確認
        content_type = response.headers.get('content-type', '').lower()
        
        # エンコーディングが指定されていない場合は自動検出
        if 'charset' not in content_type:
            # chardetで文字エンコーディングを検出
            detected = chardet.detect(response.content)
            if detected['encoding']:
                response.encoding = detected['encoding']
                self.logger.debug(f"文字エンコーディングを検出: {detected['encoding']} (信頼度: {detected['confidence']:.2f})")
            else:
                # フォールバック: 日本語サイトでよく使われるエンコーディング
                for encoding in ['utf-8', 'shift_jis', 'euc-jp', 'iso-2022-jp']:
                    try:
                        response.content.decode(encoding)
                        response.encoding = encoding
                        self.logger.debug(f"フォールバックエンコーディングを使用: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # 最終フォールバック
                    response.encoding = 'utf-8'
                    self.logger.warning("エンコーディング検出に失敗、UTF-8を使用")
        
        return response
    
    def _extract_document(self, response: requests.Response, url: str) -> Optional[Document]:
        """
        HTMLレスポンスから文書を抽出
        
        Args:
            response: HTTPレスポンス
            url: 文書のURL
            
        Returns:
            抽出された文書（抽出失敗時はNone）
        """
        try:
            soup = BeautifulSoup(response.text, 'lxml')
            
            # タイトルを抽出
            title = self._extract_title(soup, url)
            
            # メインコンテンツを抽出
            content = self._extract_content(soup)
            
            # セクション情報を抽出
            section = self._extract_section(soup, url)
            
            if not content.strip():
                self.logger.warning(f"コンテンツが空: {url}")
                return None
            
            # 文書オブジェクトを作成
            document = Document(
                title=title,
                content=content,
                url=url,
                section=section,
                timestamp=datetime.now(),
                metadata={
                    'scraped_at': datetime.now().isoformat(),
                    'content_length': len(content),
                    'encoding': response.encoding,
                    'status_code': response.status_code
                }
            )
            
            return document
            
        except Exception as e:
            self.logger.error(f"文書抽出エラー ({url}): {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """
        ページタイトルを抽出
        
        Args:
            soup: BeautifulSoupオブジェクト
            url: ページURL
            
        Returns:
            抽出されたタイトル
        """
        # <title>タグから取得
        title_tag = soup.find('title')
        if title_tag and title_tag.get_text(strip=True):
            return title_tag.get_text(strip=True)
        
        # <h1>タグから取得
        h1_tag = soup.find('h1')
        if h1_tag and h1_tag.get_text(strip=True):
            return h1_tag.get_text(strip=True)
        
        # URLからファイル名を取得
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        if path_parts and path_parts[-1]:
            return path_parts[-1].replace('.html', '').replace('.php', '')
        
        return "無題の文書"
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """
        メインコンテンツを抽出
        
        Args:
            soup: BeautifulSoupオブジェクト
            
        Returns:
            抽出されたコンテンツ
        """
        # 不要なタグを除去
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        # メインコンテンツ領域を特定
        content_selectors = [
            'main',
            '.content',
            '.main-content',
            '#content',
            '#main',
            'article',
            '.article',
            'body'
        ]
        
        content_element = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break
        
        if not content_element:
            content_element = soup.find('body') or soup
        
        # テキストを抽出
        text = content_element.get_text(separator='\n', strip=True)
        
        # 複数の改行を整理
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        content = '\n'.join(lines)
        
        return content
    
    def _extract_section(self, soup: BeautifulSoup, url: str) -> str:
        """
        セクション情報を抽出
        
        Args:
            soup: BeautifulSoupオブジェクト
            url: ページURL
            
        Returns:
            セクション名
        """
        # パンくずリストから取得
        breadcrumb = soup.select_one('.breadcrumb, .breadcrumbs, nav[aria-label="breadcrumb"]')
        if breadcrumb:
            breadcrumb_text = breadcrumb.get_text(separator=' > ', strip=True)
            if breadcrumb_text:
                return breadcrumb_text
        
        # URLパスから推測
        parsed_url = urlparse(url)
        path_parts = [part for part in parsed_url.path.strip('/').split('/') if part]
        
        if len(path_parts) > 1:
            return ' > '.join(path_parts[:-1])
        
        return "トップページ"
    
    def _extract_links(self, response: requests.Response, current_url: str) -> List[str]:
        """
        ページ内のリンクを抽出
        
        Args:
            response: HTTPレスポンス
            current_url: 現在のページURL
            
        Returns:
            同一ドメイン内のリンクのリスト
        """
        try:
            soup = BeautifulSoup(response.text, 'lxml')
            links = []
            
            base_domain = urlparse(self.base_url).netloc
            
            for link_tag in soup.find_all('a', href=True):
                href = link_tag['href']
                
                # 絶対URLに変換
                absolute_url = urljoin(current_url, href)
                parsed_url = urlparse(absolute_url)
                
                # 同一ドメインかつHTTP/HTTPSのリンクのみを対象
                if (parsed_url.netloc == base_domain and 
                    parsed_url.scheme in ['http', 'https'] and
                    not self._is_excluded_url(absolute_url)):
                    links.append(absolute_url)
            
            return list(set(links))  # 重複を除去
            
        except Exception as e:
            self.logger.error(f"リンク抽出エラー ({current_url}): {e}")
            return []
    
    def _is_excluded_url(self, url: str) -> bool:
        """
        除外対象のURLかどうかを判定
        
        Args:
            url: 判定するURL
            
        Returns:
            除外対象の場合True
        """
        excluded_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.tar', '.gz']
        excluded_patterns = ['mailto:', 'tel:', 'javascript:', '#']
        
        url_lower = url.lower()
        
        # 拡張子チェック
        for ext in excluded_extensions:
            if url_lower.endswith(ext):
                return True
        
        # パターンチェック
        for pattern in excluded_patterns:
            if pattern in url_lower:
                return True
        
        return False
    
    def extract_content(self, html: str) -> str:
        """
        HTML文字列からコンテンツを抽出（外部インターフェイス用）
        
        Args:
            html: HTML文字列
            
        Returns:
            抽出されたテキストコンテンツ
        """
        try:
            soup = BeautifulSoup(html, 'lxml')
            return self._extract_content(soup)
        except Exception as e:
            self.logger.error(f"コンテンツ抽出エラー: {e}")
            return ""
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        スクレイピング統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        return {
            'visited_urls_count': len(self.visited_urls),
            'visited_urls': list(self.visited_urls),
            'base_url': self.base_url,
            'request_delay': self.request_delay,
            'max_retries': self.max_retries,
            'timeout': self.timeout
        }
    
    def reset(self) -> None:
        """
        スクレイパーの状態をリセット
        """
        self.visited_urls.clear()
        self.logger.info("スクレイパーの状態をリセットしました")
    
    def __del__(self):
        """デストラクタ"""
        if hasattr(self, 'session'):
            self.session.close()