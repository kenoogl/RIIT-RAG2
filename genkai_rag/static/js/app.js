/**
 * 玄界RAGシステム フロントエンドJavaScript
 */

class GenkaiRAGApp {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.currentModel = null;
        this.isProcessing = false;
        
        this.initializeElements();
        this.bindEvents();
        this.loadInitialData();
        
        // 定期的にシステム状態を更新
        setInterval(() => this.updateSystemStatus(), 30000);
    }
    
    /**
     * DOM要素を初期化
     */
    initializeElements() {
        // フォーム要素
        this.questionInput = document.getElementById('questionInput');
        this.submitBtn = document.getElementById('submitBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.includeHistoryCheckbox = document.getElementById('includeHistory');
        this.maxSourcesInput = document.getElementById('maxSources');
        
        // モデル選択
        this.modelSelect = document.getElementById('modelSelect');
        this.switchModelBtn = document.getElementById('switchModelBtn');
        
        // ステータス表示
        this.systemStatus = document.getElementById('systemStatus');
        this.currentModelDisplay = document.getElementById('currentModel');
        this.activeSessions = document.getElementById('activeSessions');
        
        // 会話表示
        this.messagesContainer = document.getElementById('messagesContainer');
        this.loadingIndicator = document.getElementById('loadingIndicator');
        
        // モーダル
        this.errorModal = document.getElementById('errorModal');
        this.errorMessage = document.getElementById('errorMessage');
        this.closeErrorModal = document.getElementById('closeErrorModal');
        this.errorOkBtn = document.getElementById('errorOkBtn');
    }
    
    /**
     * イベントリスナーを設定
     */
    bindEvents() {
        // フォーム送信
        const queryForm = document.getElementById('queryForm');
        if (queryForm) {
            queryForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.submitQuery();
            });
        }
        
        this.submitBtn.addEventListener('click', (e) => {
            e.preventDefault();
            this.submitQuery();
        });
        
        this.questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.submitQuery();
            }
        });
        
        // 入力検証
        this.questionInput.addEventListener('input', () => {
            this.validateInput();
        });
        
        // 履歴クリア
        this.clearBtn.addEventListener('click', () => this.clearHistory());
        
        // モデル切り替え
        this.switchModelBtn.addEventListener('click', () => this.switchModel());
        
        // モーダル閉じる
        this.closeErrorModal.addEventListener('click', () => this.hideErrorModal());
        this.errorOkBtn.addEventListener('click', () => this.hideErrorModal());
        
        // モーダル外クリックで閉じる
        this.errorModal.addEventListener('click', (e) => {
            if (e.target === this.errorModal) {
                this.hideErrorModal();
            }
        });
        
        // キーボードナビゲーション
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.errorModal.style.display === 'flex') {
                this.hideErrorModal();
            }
        });
        
        // フォーカス管理
        this.questionInput.addEventListener('focus', () => {
            this.questionInput.setAttribute('aria-describedby', 'questionHelp');
        });
    }
    
    /**
     * 初期データを読み込み
     */
    async loadInitialData() {
        try {
            await Promise.all([
                this.loadModels(),
                this.updateSystemStatus(),
                this.loadChatHistory()
            ]);
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showError('初期データの読み込みに失敗しました');
        }
    }
    
    /**
     * 入力検証
     */
    validateInput() {
        const question = this.questionInput.value.trim();
        const isValid = question.length >= 1;
        
        this.submitBtn.disabled = !isValid || this.isProcessing;
        
        // アクセシビリティ: 入力状態をスクリーンリーダーに通知
        if (question.length === 0) {
            this.questionInput.setAttribute('aria-invalid', 'true');
            this.questionInput.setAttribute('aria-describedby', 'questionHelp questionError');
        } else {
            this.questionInput.setAttribute('aria-invalid', 'false');
            this.questionInput.setAttribute('aria-describedby', 'questionHelp');
        }
    }
    
    /**
     * フォーカス管理
     */
    manageFocus() {
        // エラーモーダル表示時のフォーカス管理
        if (this.errorModal.style.display === 'flex') {
            this.errorOkBtn.focus();
        }
    }
    
    /**
     * アクセシビリティ通知
     */
    announceToScreenReader(message) {
        // ライブリージョンを使用してスクリーンリーダーに通知
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'polite');
        announcement.setAttribute('aria-atomic', 'true');
        announcement.className = 'sr-only';
        announcement.textContent = message;
        
        document.body.appendChild(announcement);
        
        // 通知後に要素を削除
        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    }
    
    /**
     * 利用可能なモデル一覧を読み込み
     */
    async loadModels() {
        try {
            const response = await fetch('/api/models');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.populateModelSelect(data.models);
            this.currentModel = data.current_model;
            this.currentModelDisplay.textContent = this.currentModel;
            
        } catch (error) {
            console.error('Failed to load models:', error);
            this.modelSelect.innerHTML = '<option value="">エラー: モデル読み込み失敗</option>';
        }
    }
    
    /**
     * モデル選択肢を設定
     */
    populateModelSelect(models) {
        this.modelSelect.innerHTML = '';
        
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = model.display_name || model.name;
            option.selected = model.is_default;
            
            if (!model.is_available) {
                option.disabled = true;
                option.textContent += ' (利用不可)';
            }
            
            this.modelSelect.appendChild(option);
        });
    }
    
    /**
     * システム状態を更新
     */
    async updateSystemStatus() {
        try {
            const response = await fetch('/api/system/status');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const status = await response.json();
            
            // ステータス表示を更新
            this.systemStatus.textContent = this.getStatusText(status.status);
            this.systemStatus.className = `status-value ${status.status}`;
            
            this.currentModelDisplay.textContent = status.current_model;
            this.activeSessions.textContent = status.active_sessions;
            
        } catch (error) {
            console.error('Failed to update system status:', error);
            this.systemStatus.textContent = 'エラー';
            this.systemStatus.className = 'status-value unhealthy';
        }
    }
    
    /**
     * ステータステキストを取得
     */
    getStatusText(status) {
        const statusMap = {
            'healthy': '正常',
            'degraded': '低下',
            'unhealthy': '異常'
        };
        return statusMap[status] || status;
    }
    
    /**
     * チャット履歴を読み込み
     */
    async loadChatHistory() {
        try {
            const response = await fetch(`/api/chat/history?session_id=${this.sessionId}&limit=20`);
            if (!response.ok) {
                if (response.status === 404) {
                    // 履歴がない場合は正常
                    return;
                }
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.displayMessages(data.messages);
            
        } catch (error) {
            console.error('Failed to load chat history:', error);
            // 履歴読み込み失敗は致命的ではないのでエラーモーダルは表示しない
        }
    }
    
    /**
     * 質問を送信
     */
    async submitQuery() {
        const question = this.questionInput.value.trim();
        if (!question) {
            this.showError('質問を入力してください');
            this.questionInput.focus();
            return;
        }
        
        if (this.isProcessing) {
            return;
        }
        
        this.setProcessingState(true);
        
        try {
            // ユーザーメッセージを即座に表示
            this.addMessage('user', question);
            
            // APIリクエストを送信
            const requestData = {
                question: question,
                session_id: this.sessionId,
                model_name: this.modelSelect.value || null,
                max_sources: parseInt(this.maxSourcesInput.value) || 5,
                include_history: this.includeHistoryCheckbox.checked
            };
            
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // アシスタントの回答を表示
            this.addMessage('assistant', data.answer, data.sources, {
                processing_time: data.processing_time,
                model_used: data.model_used
            });
            
            // 入力フィールドをクリア
            this.questionInput.value = '';
            
            // 成功をスクリーンリーダーに通知
            this.announceToScreenReader('回答が生成されました。');
            
        } catch (error) {
            console.error('Failed to submit query:', error);
            this.showError(`質問の処理に失敗しました: ${error.message}`);
            
            // エラーメッセージを表示
            this.addMessage('assistant', 'エラーが発生しました。もう一度お試しください。', [], {
                error: true
            });
            
            // エラーをスクリーンリーダーに通知
            this.announceToScreenReader('エラーが発生しました。もう一度お試しください。');
            
        } finally {
            this.setProcessingState(false);
        }
    }
    
    /**
     * セッションIDを生成
     */
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    /**
     * 処理状態を設定
     */
    setProcessingState(processing) {
        this.isProcessing = processing;
        
        if (processing) {
            this.submitBtn.disabled = true;
            this.submitBtn.textContent = '処理中...';
            this.submitBtn.setAttribute('aria-busy', 'true');
            this.loadingIndicator.style.display = 'block';
            this.loadingIndicator.setAttribute('aria-hidden', 'false');
            
            // スクリーンリーダーに処理開始を通知
            this.announceToScreenReader('質問を処理中です。しばらくお待ちください。');
        } else {
            this.submitBtn.disabled = false;
            this.submitBtn.textContent = '質問する';
            this.submitBtn.setAttribute('aria-busy', 'false');
            this.loadingIndicator.style.display = 'none';
            this.loadingIndicator.setAttribute('aria-hidden', 'true');
            
            // 入力検証を再実行
            this.validateInput();
        }
    }
    
    /**
     * メッセージを追加
     */
    addMessage(role, content, sources = [], metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const headerDiv = document.createElement('div');
        headerDiv.className = 'message-header';
        
        const roleSpan = document.createElement('span');
        roleSpan.className = 'message-role';
        roleSpan.textContent = role === 'user' ? 'ユーザー' : 'アシスタント';
        
        const timeSpan = document.createElement('span');
        timeSpan.className = 'message-time';
        timeSpan.textContent = new Date().toLocaleString('ja-JP');
        
        headerDiv.appendChild(roleSpan);
        headerDiv.appendChild(timeSpan);
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        messageDiv.appendChild(headerDiv);
        messageDiv.appendChild(contentDiv);
        
        // 出典情報を追加
        if (sources && sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'message-sources';
            
            const sourcesTitle = document.createElement('div');
            sourcesTitle.className = 'sources-title';
            sourcesTitle.textContent = '出典:';
            sourcesDiv.appendChild(sourcesTitle);
            
            sources.forEach(source => {
                const sourceItem = document.createElement('div');
                sourceItem.className = 'source-item';
                
                const sourceLink = document.createElement('a');
                sourceLink.className = 'source-url';
                sourceLink.href = source.url;
                sourceLink.target = '_blank';
                sourceLink.textContent = source.url;
                
                const sourceTitle = document.createElement('span');
                sourceTitle.className = 'source-title';
                sourceTitle.textContent = source.title || '';
                
                sourceItem.appendChild(sourceLink);
                if (source.title) {
                    sourceItem.appendChild(sourceTitle);
                }
                
                sourcesDiv.appendChild(sourceItem);
            });
            
            messageDiv.appendChild(sourcesDiv);
        }
        
        // メタデータ情報を追加（デバッグ用）
        if (metadata.processing_time) {
            const metaDiv = document.createElement('div');
            metaDiv.style.fontSize = '0.8rem';
            metaDiv.style.color = '#666';
            metaDiv.style.marginTop = '10px';
            metaDiv.textContent = `処理時間: ${metadata.processing_time.toFixed(2)}秒`;
            if (metadata.model_used) {
                metaDiv.textContent += ` | モデル: ${metadata.model_used}`;
            }
            messageDiv.appendChild(metaDiv);
        }
        
        this.messagesContainer.appendChild(messageDiv);
        
        // スクロールを最下部に移動
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    /**
     * メッセージ一覧を表示
     */
    displayMessages(messages) {
        this.messagesContainer.innerHTML = '';
        
        messages.forEach(msg => {
            const role = msg.role === 'user' ? 'user' : 'assistant';
            const sources = msg.metadata?.sources || [];
            this.addMessage(role, msg.content, sources);
        });
    }
    
    /**
     * 履歴をクリア
     */
    async clearHistory() {
        if (!confirm('会話履歴をクリアしますか？')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/chat/history/${this.sessionId}`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            // 表示をクリア
            this.messagesContainer.innerHTML = '';
            
            // 新しいセッションIDを生成
            this.sessionId = this.generateSessionId();
            
        } catch (error) {
            console.error('Failed to clear history:', error);
            this.showError('履歴のクリアに失敗しました');
        }
    }
    
    /**
     * モデルを切り替え
     */
    async switchModel() {
        const selectedModel = this.modelSelect.value;
        if (!selectedModel) {
            this.showError('モデルを選択してください');
            return;
        }
        
        if (selectedModel === this.currentModel) {
            this.showError('既に選択されているモデルです');
            return;
        }
        
        try {
            this.switchModelBtn.disabled = true;
            this.switchModelBtn.textContent = '切り替え中...';
            
            const response = await fetch('/api/models/switch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_name: selectedModel,
                    force: false
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.currentModel = selectedModel;
                this.currentModelDisplay.textContent = selectedModel;
                this.addMessage('assistant', `モデルを ${selectedModel} に切り替えました。`, [], {
                    system: true
                });
            } else {
                throw new Error(data.message || 'モデルの切り替えに失敗しました');
            }
            
        } catch (error) {
            console.error('Failed to switch model:', error);
            this.showError(`モデルの切り替えに失敗しました: ${error.message}`);
            
        } finally {
            this.switchModelBtn.disabled = false;
            this.switchModelBtn.textContent = 'モデル切り替え';
        }
    }
    
    /**
     * エラーモーダルを表示
     */
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorModal.style.display = 'flex';
        this.errorModal.setAttribute('aria-hidden', 'false');
        
        // フォーカスをモーダルに移動
        this.manageFocus();
        
        // スクリーンリーダーに通知
        this.announceToScreenReader(`エラー: ${message}`);
    }
    
    /**
     * エラーモーダルを非表示
     */
    hideErrorModal() {
        this.errorModal.style.display = 'none';
        this.errorModal.setAttribute('aria-hidden', 'true');
        
        // フォーカスを質問入力フィールドに戻す
        this.questionInput.focus();
    }
}

// アプリケーションを初期化
document.addEventListener('DOMContentLoaded', () => {
    window.genkaiApp = new GenkaiRAGApp();
});