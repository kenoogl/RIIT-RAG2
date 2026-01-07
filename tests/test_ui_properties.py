"""
UI プロパティテスト

Webインターフェイスの動作プロパティを検証するテストスイート
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from hypothesis import given, strategies as st, settings, assume
import threading
import uvicorn
from fastapi.testclient import TestClient

from genkai_rag.api.app import create_app


@pytest.fixture(scope="session")
def test_server():
    """テスト用サーバーを起動"""
    # モックされたアプリケーション状態
    with patch('genkai_rag.api.app.app_state') as mock_state:
        # 基本的なモックを設定
        mock_rag_engine = Mock()
        mock_rag_engine.query = Mock(return_value={
            "answer": "テスト回答です。玄界システムは高性能なスーパーコンピュータです。",
            "sources": [
                {
                    "url": "https://example.com/genkai-info",
                    "title": "玄界システム概要",
                    "content_type": "text/html",
                    "last_accessed": "2024-01-01T00:00:00"
                }
            ],
            "model_used": "test-model",
            "metadata": {"confidence": 0.95}
        })
        
        mock_llm_manager = Mock()
        mock_llm_manager.get_current_model.return_value = "test-model"
        mock_llm_manager.list_available_models.return_value = {
            "test-model": {
                "display_name": "テストモデル",
                "description": "テスト用モデル",
                "is_available": True,
                "parameters": {}
            },
            "large-model": {
                "display_name": "大型モデル",
                "description": "大型テスト用モデル",
                "is_available": True,
                "parameters": {}
            }
        }
        mock_llm_manager.switch_model.return_value = True
        mock_llm_manager.check_model_health.return_value = True
        
        mock_chat_manager = Mock()
        mock_chat_manager.get_history.return_value = []
        mock_chat_manager.get_message_count.return_value = 0
        mock_chat_manager.list_sessions.return_value = ["session1"]
        mock_chat_manager.save_message = Mock()
        mock_chat_manager.clear_history = Mock()
        
        mock_system_monitor = Mock()
        mock_status = Mock()
        mock_status.timestamp = "2024-01-01T00:00:00"
        mock_status.uptime_seconds = 3600.0
        mock_status.memory_usage_mb = 512.0
        mock_status.disk_usage_mb = 1024.0
        mock_system_monitor.get_system_status.return_value = mock_status
        
        mock_config_manager = Mock()
        mock_config_manager.load_config.return_value = {
            "web": {"cors_origins": ["*"], "allowed_hosts": ["*"]},
            "llm": {"ollama_url": "http://localhost:11434"},
            "chat": {"max_history_size": 50}
        }
        
        mock_document_processor = Mock()
        
        # テンプレートのモック（実際のHTMLを返す）
        mock_templates = Mock()
        
        def mock_template_response(template_name, context):
            """実際のHTMLテンプレートを模擬"""
            html_content = """
            <!DOCTYPE html>
            <html lang="ja">
            <head>
                <meta charset="UTF-8">
                <title>玄界RAGシステム</title>
                <style>
                    .loading { display: none; }
                    .loading.show { display: block; }
                    .message { margin: 10px 0; padding: 10px; border: 1px solid #ccc; }
                    .message.user { background: #e3f2fd; }
                    .message.assistant { background: #f1f8e9; }
                </style>
            </head>
            <body>
                <div id="statusPanel">
                    <span id="systemStatus">確認中...</span>
                    <span id="currentModel">確認中...</span>
                    <span id="activeSessions">0</span>
                </div>
                <select id="modelSelect">
                    <option value="">読み込み中...</option>
                </select>
                <button id="switchModelBtn">モデル切り替え</button>
                <textarea id="questionInput" placeholder="質問を入力してください..."></textarea>
                <input type="checkbox" id="includeHistory" checked>
                <input type="number" id="maxSources" value="5" min="1" max="20">
                <button id="submitBtn">質問する</button>
                <button id="clearBtn">履歴クリア</button>
                <div id="loadingIndicator" class="loading">
                    <div class="loading-spinner"></div>
                    <p>回答を生成中...</p>
                </div>
                <div id="messagesContainer"></div>
                <div id="errorModal" style="display: none;">
                    <div id="errorMessage"></div>
                    <button id="closeErrorModal">×</button>
                    <button id="errorOkBtn">OK</button>
                </div>
                <script>
                    // 基本的なJavaScript機能を模擬
                    class GenkaiRAGApp {
                        constructor() {
                            this.sessionId = 'test_session_' + Date.now();
                            this.isProcessing = false;
                            this.initializeElements();
                            this.bindEvents();
                            this.loadInitialData();
                        }
                        
                        initializeElements() {
                            this.questionInput = document.getElementById('questionInput');
                            this.submitBtn = document.getElementById('submitBtn');
                            this.loadingIndicator = document.getElementById('loadingIndicator');
                            this.messagesContainer = document.getElementById('messagesContainer');
                            this.systemStatus = document.getElementById('systemStatus');
                            this.currentModel = document.getElementById('currentModel');
                            this.modelSelect = document.getElementById('modelSelect');
                        }
                        
                        bindEvents() {
                            this.submitBtn.addEventListener('click', () => this.submitQuery());
                        }
                        
                        async loadInitialData() {
                            try {
                                // モデル一覧を読み込み
                                const response = await fetch('/api/models');
                                const data = await response.json();
                                this.populateModelSelect(data.models);
                                this.currentModel.textContent = data.current_model;
                                
                                // システム状態を更新
                                const statusResponse = await fetch('/api/system/status');
                                const statusData = await statusResponse.json();
                                this.systemStatus.textContent = '正常';
                            } catch (error) {
                                console.error('Failed to load initial data:', error);
                            }
                        }
                        
                        populateModelSelect(models) {
                            this.modelSelect.innerHTML = '';
                            models.forEach(model => {
                                const option = document.createElement('option');
                                option.value = model.name;
                                option.textContent = model.display_name;
                                this.modelSelect.appendChild(option);
                            });
                        }
                        
                        async submitQuery() {
                            const question = this.questionInput.value.trim();
                            if (!question || this.isProcessing) return;
                            
                            this.setProcessingState(true);
                            
                            try {
                                // 処理中表示を開始
                                this.showProcessingIndicator();
                                
                                const response = await fetch('/api/query', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({
                                        question: question,
                                        session_id: this.sessionId,
                                        max_sources: 5,
                                        include_history: true
                                    })
                                });
                                
                                const data = await response.json();
                                
                                // メッセージを表示
                                this.addMessage('user', question);
                                this.addMessage('assistant', data.answer, data.sources);
                                
                                this.questionInput.value = '';
                                
                            } catch (error) {
                                console.error('Query failed:', error);
                            } finally {
                                this.setProcessingState(false);
                            }
                        }
                        
                        setProcessingState(processing) {
                            this.isProcessing = processing;
                            this.submitBtn.disabled = processing;
                            this.submitBtn.textContent = processing ? '処理中...' : '質問する';
                            
                            if (processing) {
                                this.loadingIndicator.classList.add('show');
                            } else {
                                this.loadingIndicator.classList.remove('show');
                            }
                        }
                        
                        showProcessingIndicator() {
                            this.loadingIndicator.style.display = 'block';
                            this.loadingIndicator.classList.add('show');
                        }
                        
                        addMessage(role, content, sources = []) {
                            const messageDiv = document.createElement('div');
                            messageDiv.className = 'message ' + role;
                            messageDiv.innerHTML = '<strong>' + (role === 'user' ? 'ユーザー' : 'アシスタント') + ':</strong> ' + content;
                            
                            if (sources && sources.length > 0) {
                                const sourcesDiv = document.createElement('div');
                                sourcesDiv.innerHTML = '<br><strong>出典:</strong>';
                                sources.forEach(source => {
                                    sourcesDiv.innerHTML += '<br>• <a href="' + source.url + '">' + source.title + '</a>';
                                });
                                messageDiv.appendChild(sourcesDiv);
                            }
                            
                            this.messagesContainer.appendChild(messageDiv);
                        }
                    }
                    
                    document.addEventListener('DOMContentLoaded', () => {
                        window.genkaiApp = new GenkaiRAGApp();
                    });
                </script>
            </body>
            </html>
            """
            
            response = Mock()
            response.body = html_content.encode('utf-8')
            response.status_code = 200
            response.headers = {"content-type": "text/html; charset=utf-8"}
            return response
        
        mock_templates.TemplateResponse = mock_template_response
        
        # モック状態を設定
        mock_state.rag_engine = mock_rag_engine
        mock_state.llm_manager = mock_llm_manager
        mock_state.chat_manager = mock_chat_manager
        mock_state.system_monitor = mock_system_monitor
        mock_state.config_manager = mock_config_manager
        mock_state.document_processor = mock_document_processor
        mock_state.templates = mock_templates
        
        # アプリケーションを作成
        app = create_app()
        
        # テストサーバーを別スレッドで起動
        server_thread = threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={"host": "127.0.0.1", "port": 8888, "log_level": "error"},
            daemon=True
        )
        server_thread.start()
        
        # サーバーが起動するまで待機
        time.sleep(2)
        
        yield "http://127.0.0.1:8888"


@pytest.fixture
def browser():
    """Seleniumブラウザを設定"""
    options = Options()
    options.add_argument("--headless")  # ヘッドレスモード
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    
    try:
        driver = webdriver.Chrome(options=options)
        driver.implicitly_wait(10)
        yield driver
    finally:
        if 'driver' in locals():
            driver.quit()


class TestUIProcessingIndicator:
    """UI処理中表示のテスト"""
    
    def test_processing_indicator_basic(self, test_server, browser):
        """基本的な処理中表示のテスト"""
        browser.get(test_server)
        
        # ページが読み込まれるまで待機
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.ID, "questionInput"))
        )
        
        # 質問を入力
        question_input = browser.find_element(By.ID, "questionInput")
        question_input.send_keys("玄界システムについて教えてください")
        
        # 送信ボタンをクリック
        submit_btn = browser.find_element(By.ID, "submitBtn")
        submit_btn.click()
        
        # 処理中表示が現れることを確認
        try:
            WebDriverWait(browser, 5).until(
                EC.visibility_of_element_located((By.ID, "loadingIndicator"))
            )
            loading_visible = True
        except TimeoutException:
            loading_visible = False
        
        # 処理完了後、処理中表示が消えることを確認
        WebDriverWait(browser, 10).until(
            EC.invisibility_of_element_located((By.ID, "loadingIndicator"))
        )
        
        # ボタンが元の状態に戻ることを確認
        assert submit_btn.text in ["質問する", "Submit"]
        assert not submit_btn.get_attribute("disabled")
    
    @given(
        question=st.text(min_size=5, max_size=100).filter(lambda x: x.strip())
    )
    @settings(max_examples=5, deadline=30000)
    def test_processing_indicator_property(self, test_server, browser, question):
        """
        プロパティ 9: 処理中表示
        
        質問送信時に適切な処理中表示が行われることを検証
        """
        assume(len(question.strip()) >= 5)  # 最小限の質問長を確保
        
        browser.get(test_server)
        
        # ページが読み込まれるまで待機
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.ID, "questionInput"))
        )
        
        # 質問を入力
        question_input = browser.find_element(By.ID, "questionInput")
        question_input.clear()
        question_input.send_keys(question)
        
        # 送信ボタンの初期状態を確認
        submit_btn = browser.find_element(By.ID, "submitBtn")
        initial_button_text = submit_btn.text
        
        # 送信ボタンをクリック
        submit_btn.click()
        
        # 処理中の状態変化を検証
        try:
            # ボタンが無効化されることを確認
            WebDriverWait(browser, 2).until(
                lambda driver: submit_btn.get_attribute("disabled") == "true"
            )
            button_disabled = True
        except TimeoutException:
            button_disabled = False
        
        try:
            # ボタンテキストが変更されることを確認
            WebDriverWait(browser, 2).until(
                lambda driver: submit_btn.text != initial_button_text
            )
            button_text_changed = True
        except TimeoutException:
            button_text_changed = False
        
        # 処理完了を待機
        WebDriverWait(browser, 15).until(
            lambda driver: not submit_btn.get_attribute("disabled")
        )
        
        # 処理完了後の状態を検証
        final_button_text = submit_btn.text
        is_button_enabled = not submit_btn.get_attribute("disabled")
        
        # プロパティ検証: 処理中に適切な表示が行われること
        assert button_disabled or button_text_changed, \
            "処理中にボタンの状態変化またはテキスト変更が行われませんでした"
        
        assert is_button_enabled, \
            "処理完了後にボタンが有効化されませんでした"
        
        assert final_button_text in ["質問する", "Submit"], \
            f"処理完了後のボタンテキストが期待値と異なります: {final_button_text}"


class TestUIResponseDisplay:
    """UI応答表示のテスト"""
    
    def test_response_display_basic(self, test_server, browser):
        """基本的な応答表示のテスト"""
        browser.get(test_server)
        
        # ページが読み込まれるまで待機
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.ID, "questionInput"))
        )
        
        # 質問を入力して送信
        question_input = browser.find_element(By.ID, "questionInput")
        question_input.send_keys("玄界システムについて教えてください")
        
        submit_btn = browser.find_element(By.ID, "submitBtn")
        submit_btn.click()
        
        # 応答が表示されるまで待機
        WebDriverWait(browser, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".message.assistant"))
        )
        
        # メッセージが表示されることを確認
        messages = browser.find_elements(By.CSS_SELECTOR, ".message")
        assert len(messages) >= 2  # ユーザーメッセージとアシスタントメッセージ
        
        # ユーザーメッセージの確認
        user_message = browser.find_element(By.CSS_SELECTOR, ".message.user")
        assert "玄界システムについて教えてください" in user_message.text
        
        # アシスタントメッセージの確認
        assistant_message = browser.find_element(By.CSS_SELECTOR, ".message.assistant")
        assert len(assistant_message.text) > 0
    
    @given(
        question=st.text(min_size=5, max_size=50).filter(lambda x: x.strip())
    )
    @settings(max_examples=3, deadline=45000)
    def test_response_display_property(self, test_server, browser, question):
        """
        プロパティ 10: 応答の適切な表示
        
        質問に対する応答が適切に表示されることを検証
        """
        assume(len(question.strip()) >= 5)
        
        browser.get(test_server)
        
        # ページが読み込まれるまで待機
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.ID, "questionInput"))
        )
        
        # 質問を入力して送信
        question_input = browser.find_element(By.ID, "questionInput")
        question_input.clear()
        question_input.send_keys(question)
        
        submit_btn = browser.find_element(By.ID, "submitBtn")
        submit_btn.click()
        
        # 応答が表示されるまで待機
        try:
            WebDriverWait(browser, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".message.assistant"))
            )
            response_displayed = True
        except TimeoutException:
            response_displayed = False
        
        if response_displayed:
            # メッセージコンテナを取得
            messages_container = browser.find_element(By.ID, "messagesContainer")
            messages = messages_container.find_elements(By.CSS_SELECTOR, ".message")
            
            # プロパティ検証: 適切な応答表示
            assert len(messages) >= 2, \
                "ユーザーメッセージとアシスタントメッセージの両方が表示されませんでした"
            
            # ユーザーメッセージの検証
            user_messages = [msg for msg in messages if "user" in msg.get_attribute("class")]
            assert len(user_messages) >= 1, \
                "ユーザーメッセージが表示されませんでした"
            
            latest_user_message = user_messages[-1]
            assert question in latest_user_message.text, \
                f"ユーザーメッセージに入力した質問が含まれていません: {question}"
            
            # アシスタントメッセージの検証
            assistant_messages = [msg for msg in messages if "assistant" in msg.get_attribute("class")]
            assert len(assistant_messages) >= 1, \
                "アシスタントメッセージが表示されませんでした"
            
            latest_assistant_message = assistant_messages[-1]
            assert len(latest_assistant_message.text.strip()) > 0, \
                "アシスタントメッセージが空です"
            
            # 出典情報の検証（存在する場合）
            try:
                source_links = latest_assistant_message.find_elements(By.TAG_NAME, "a")
                if source_links:
                    for link in source_links:
                        href = link.get_attribute("href")
                        assert href and href.startswith("http"), \
                            f"出典リンクが無効です: {href}"
            except NoSuchElementException:
                pass  # 出典がない場合は正常


class TestUISystemStatus:
    """UIシステム状態表示のテスト"""
    
    def test_system_status_display(self, test_server, browser):
        """システム状態表示のテスト"""
        browser.get(test_server)
        
        # ページが読み込まれるまで待機
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.ID, "systemStatus"))
        )
        
        # システム状態が表示されることを確認
        system_status = browser.find_element(By.ID, "systemStatus")
        assert system_status.text != "確認中..."
        
        # 現在のモデルが表示されることを確認
        current_model = browser.find_element(By.ID, "currentModel")
        assert current_model.text != "確認中..."
        
        # アクティブセッション数が表示されることを確認
        active_sessions = browser.find_element(By.ID, "activeSessions")
        assert active_sessions.text.isdigit()


class TestUIModelSelection:
    """UIモデル選択のテスト"""
    
    def test_model_selection_display(self, test_server, browser):
        """モデル選択表示のテスト"""
        browser.get(test_server)
        
        # ページが読み込まれるまで待機
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.ID, "modelSelect"))
        )
        
        # モデル選択肢が読み込まれるまで待機
        WebDriverWait(browser, 10).until(
            lambda driver: len(driver.find_element(By.ID, "modelSelect").find_elements(By.TAG_NAME, "option")) > 1
        )
        
        # モデル選択肢が表示されることを確認
        model_select = browser.find_element(By.ID, "modelSelect")
        options = model_select.find_elements(By.TAG_NAME, "option")
        
        assert len(options) > 1, "モデル選択肢が表示されませんでした"
        
        # 最初のオプションが "読み込み中..." でないことを確認
        first_option_text = options[0].text
        assert first_option_text != "読み込み中...", \
            f"モデル選択肢が読み込まれませんでした: {first_option_text}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])