"""Webスクレイピング機能のテスト

Feature: genkai-rag-system, Property 1: Webスクレイピング機能
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from unittest.mock import Mock, patch, MagicMock
import requests
from bs4 import BeautifulSoup

from genkai_rag.core.scraper import WebScraper
from genkai_rag.models.document import Document

# Hypothesisの設定：タイムアウトを無効化し、テスト実行時間を短縮
settings.register_profile("test", deadline=None, max_examples=10)
settings.load_profile("test")


class TestWebScraper:
    """WebScraperクラスの基本テスト"""
    
    def test_scraper_initialization(self):
        """スクレイパーの初期化テスト"""
        scraper = WebScraper(
            base_url="https://example.com",
            request_delay=0.5,
            max_retries=2
        )
        
        assert scraper.base_url == "https://example.com"
        assert scraper.request_delay == 0.5
        assert scraper.max_retries == 2
        assert len(scraper.visited_urls) == 0
    
    def test_japanese_encoding_detection(self):
        """日本語エンコーディング検出テスト"""
        scraper = WebScraper()
        
        # モックレスポンスを作成
        mock_response = Mock()
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.content = "テストコンテンツ".encode('utf-8')
        
        # エンコーディング処理をテスト
        processed_response = scraper.handle_japanese_encoding(mock_response)
        
        assert processed_response.encoding in ['utf-8', 'shift_jis', 'euc-jp', 'iso-2022-jp']
    
    def test_content_extraction(self):
        """コンテンツ抽出テスト"""
        scraper = WebScraper()
        
        html = """
        <html>
        <head><title>テストページ</title></head>
        <body>
            <nav>ナビゲーション</nav>
            <main>
                <h1>メインタイトル</h1>
                <p>これはテストコンテンツです。</p>
                <p>複数の段落があります。</p>
            </main>
            <footer>フッター</footer>
        </body>
        </html>
        """
        
        content = scraper.extract_content(html)
        
        assert "メインタイトル" in content
        assert "これはテストコンテンツです。" in content
        assert "複数の段落があります。" in content
        # ナビゲーションとフッターは除外される
        assert "ナビゲーション" not in content
        assert "フッター" not in content
    
    @patch('requests.Session.get')
    def test_single_page_scraping_success(self, mock_get):
        """単一ページスクレイピング成功テスト"""
        # モックレスポンスを設定
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html; charset=utf-8'}
        mock_response.encoding = 'utf-8'
        mock_response.text = """
        <html>
        <head><title>テストページ</title></head>
        <body>
            <main>
                <h1>テストタイトル</h1>
                <p>テストコンテンツです。</p>
            </main>
        </body>
        </html>
        """
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        scraper = WebScraper()
        document = scraper.scrape_single_page("https://example.com/test")
        
        assert document is not None
        assert document.title == "テストページ"
        assert "テストコンテンツです。" in document.content
        assert document.url == "https://example.com/test"
        assert document.is_valid()
    
    @patch('requests.Session.get')
    def test_single_page_scraping_failure(self, mock_get):
        """単一ページスクレイピング失敗テスト"""
        # リクエスト失敗をシミュレート
        mock_get.side_effect = requests.exceptions.RequestException("接続エラー")
        
        scraper = WebScraper(max_retries=1)
        document = scraper.scrape_single_page("https://example.com/test")
        
        assert document is None
    
    def test_url_exclusion(self):
        """URL除外機能テスト"""
        scraper = WebScraper()
        
        # 除外対象のURL
        assert scraper._is_excluded_url("https://example.com/file.pdf")
        assert scraper._is_excluded_url("https://example.com/doc.docx")
        assert scraper._is_excluded_url("mailto:test@example.com")
        assert scraper._is_excluded_url("javascript:void(0)")
        
        # 除外対象でないURL
        assert not scraper._is_excluded_url("https://example.com/page.html")
        assert not scraper._is_excluded_url("https://example.com/info")
    
    def test_statistics_and_reset(self):
        """統計情報とリセット機能テスト"""
        scraper = WebScraper()
        
        # 訪問済みURLを追加
        scraper.visited_urls.add("https://example.com/page1")
        scraper.visited_urls.add("https://example.com/page2")
        
        stats = scraper.get_statistics()
        assert stats['visited_urls_count'] == 2
        assert "https://example.com/page1" in stats['visited_urls']
        
        # リセット
        scraper.reset()
        assert len(scraper.visited_urls) == 0


class TestWebScrapingProperties:
    """Webスクレイピング機能のプロパティテスト
    
    Feature: genkai-rag-system, Property 1: Webスクレイピング機能
    """
    
    @settings(deadline=None, max_examples=5)
    @given(
        title=st.text(min_size=1, max_size=100, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
        content=st.text(min_size=1, max_size=5000, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
        url=st.text(min_size=10, max_size=200, alphabet=st.characters(min_codepoint=32, max_codepoint=126))
    )
    def test_html_content_extraction_properties(self, title, content, url):
        """
        プロパティ 1: Webスクレイピング機能
        任意のHTMLコンテンツに対して、スクレイピングシステムは適切に
        テキストコンテンツを抽出し、有効な文書オブジェクトを生成する
        
        Feature: genkai-rag-system, Property 1: Webスクレイピング機能
        """
        # 前提条件
        assume(len(title.strip()) > 0)
        assume(len(content.strip()) > 0)
        assume(len(url.strip()) > 0)
        assume('</script>' not in content)  # スクリプトタグを避ける
        assume('</style>' not in content)   # スタイルタグを避ける
        assume('<' not in content)  # HTMLタグを避ける
        assume('>' not in content)  # HTMLタグを避ける
        
        scraper = WebScraper()
        
        # 有効なHTMLを生成
        html = f"""
        <html>
        <head><title>{title}</title></head>
        <body>
            <main>
                <h1>{title}</h1>
                <p>{content}</p>
            </main>
        </body>
        </html>
        """
        
        # コンテンツ抽出をテスト
        extracted_content = scraper.extract_content(html)
        
        # プロパティ1: 抽出されたコンテンツが空でない
        assert len(extracted_content.strip()) > 0
        
        # プロパティ2: 元のコンテンツが含まれている
        assert title in extracted_content or content in extracted_content
        
        # プロパティ3: HTMLタグが除去されている（基本的なチェック）
        # 完全なHTMLタグの除去は保証しないが、主要なタグは除去される
        assert '<html>' not in extracted_content
        assert '<body>' not in extracted_content
        assert '<main>' not in extracted_content
    
    @settings(deadline=None, max_examples=5)
    @patch('requests.Session.get')
    @given(
        title=st.text(min_size=1, max_size=50, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
        content=st.text(min_size=1, max_size=1000, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
        status_code=st.sampled_from([200, 201, 202])
    )
    def test_successful_page_scraping_properties(self, mock_get, title, content, status_code):
        """
        成功したページスクレイピングのプロパティテスト
        
        Feature: genkai-rag-system, Property 1: Webスクレイピング機能
        """
        # 前提条件
        assume(len(title.strip()) > 0)
        assume(len(content.strip()) > 0)
        assume('<' not in title)  # HTMLタグを避ける
        assume('<' not in content)
        
        # モックレスポンスを設定
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.headers = {'content-type': 'text/html; charset=utf-8'}
        mock_response.encoding = 'utf-8'
        mock_response.text = f"""
        <html>
        <head><title>{title}</title></head>
        <body>
            <main>
                <h1>{title}</h1>
                <p>{content}</p>
            </main>
        </body>
        </html>
        """
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        scraper = WebScraper()
        test_url = "https://example.com/test"
        document = scraper.scrape_single_page(test_url)
        
        # プロパティ1: 有効な文書が生成される
        assert document is not None
        assert document.is_valid()
        
        # プロパティ2: 文書の基本情報が正しく設定される
        assert document.title == title
        assert document.url == test_url
        assert len(document.content.strip()) > 0
        
        # プロパティ3: 元のコンテンツが保持される
        assert title in document.content or content in document.content
        
        # プロパティ4: メタデータが設定される
        assert 'scraped_at' in document.metadata
        assert 'content_length' in document.metadata
        assert document.metadata['status_code'] == status_code
    
    @settings(deadline=None, max_examples=3)
    @patch('requests.Session.get')
    @given(
        error_type=st.sampled_from([
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError
        ]),
        max_retries=st.integers(min_value=1, max_value=3)
    )
    def test_error_handling_properties(self, mock_get, error_type, max_retries):
        """
        エラーハンドリングのプロパティテスト
        
        Feature: genkai-rag-system, Property 1: Webスクレイピング機能
        """
        # リクエストエラーをシミュレート
        mock_get.side_effect = error_type("テストエラー")
        
        scraper = WebScraper(max_retries=max_retries)
        document = scraper.scrape_single_page("https://example.com/test")
        
        # プロパティ1: エラー時はNoneを返す
        assert document is None
        
        # プロパティ2: リトライが実行される（正確な回数は実装依存）
        assert mock_get.call_count >= max_retries
    
    @settings(deadline=None, max_examples=3, suppress_health_check=[HealthCheck.filter_too_much])
    @given(
        base_url=st.sampled_from(["https://example.com", "http://test.org", "https://site.net"]),
        request_delay=st.floats(min_value=0.0, max_value=1.0),
        max_retries=st.integers(min_value=0, max_value=3),
        timeout=st.integers(min_value=5, max_value=30)
    )
    def test_scraper_configuration_properties(self, base_url, request_delay, max_retries, timeout):
        """
        スクレイパー設定のプロパティテスト
        
        Feature: genkai-rag-system, Property 1: Webスクレイピング機能
        """
        scraper = WebScraper(
            base_url=base_url,
            request_delay=request_delay,
            max_retries=max_retries,
            timeout=timeout
        )
        
        # プロパティ1: 設定値が正しく保存される
        assert scraper.base_url == base_url
        assert scraper.request_delay == request_delay
        assert scraper.max_retries == max_retries
        assert scraper.timeout == timeout
        
        # プロパティ2: 初期状態が正しい
        assert len(scraper.visited_urls) == 0
        assert scraper.session is not None
        
        # プロパティ3: 統計情報が正しい
        stats = scraper.get_statistics()
        assert stats['base_url'] == base_url
        assert stats['request_delay'] == request_delay
        assert stats['max_retries'] == max_retries
        assert stats['visited_urls_count'] == 0
    
    def test_link_extraction_properties(self):
        """
        リンク抽出のプロパティテスト
        
        Feature: genkai-rag-system, Property 1: Webスクレイピング機能
        """
        scraper = WebScraper(base_url="https://example.com")
        
        # テスト用HTMLレスポンス
        mock_response = Mock()
        mock_response.text = """
        <html>
        <body>
            <a href="/page1.html">ページ1</a>
            <a href="https://example.com/page2.html">ページ2</a>
            <a href="https://other.com/page3.html">外部ページ</a>
            <a href="mailto:test@example.com">メール</a>
            <a href="/document.pdf">PDF</a>
            <a href="javascript:void(0)">JavaScript</a>
        </body>
        </html>
        """
        
        current_url = "https://example.com/current"
        links = scraper._extract_links(mock_response, current_url)
        
        # プロパティ1: 同一ドメインのHTMLページのみが抽出される
        valid_links = [link for link in links if 'example.com' in link and link.endswith('.html')]
        assert len(valid_links) >= 1
        
        # プロパティ2: 除外対象のリンクは含まれない
        for link in links:
            assert 'mailto:' not in link
            assert 'javascript:' not in link
            assert not link.endswith('.pdf')
        
        # プロパティ3: 重複が除去されている
        assert len(links) == len(set(links))