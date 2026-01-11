// 最小限のテストJavaScript
console.log('test.js loaded');

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded in test.js');
    
    // 基本的なDOM要素の確認
    const messagesContainer = document.getElementById('messagesContainer');
    console.log('messagesContainer found:', !!messagesContainer);
    
    const submitBtn = document.getElementById('submitBtn');
    console.log('submitBtn found:', !!submitBtn);
    
    if (submitBtn) {
        submitBtn.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('Submit button clicked!');
            
            if (messagesContainer) {
                const testDiv = document.createElement('div');
                testDiv.textContent = 'テストメッセージ - ' + new Date().toLocaleString();
                testDiv.style.padding = '10px';
                testDiv.style.border = '1px solid #ccc';
                testDiv.style.margin = '5px 0';
                messagesContainer.appendChild(testDiv);
                console.log('Test message added');
            } else {
                console.error('messagesContainer not found when trying to add message');
            }
        });
    }
});