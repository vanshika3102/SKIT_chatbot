// static/js/chat_script.js

document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.getElementById('chat-container');
    const form = document.getElementById('input-form');
    const input = document.getElementById('question-input');
    const submitBtn = document.getElementById('submit-btn');
    const clearChatBtn = document.getElementById('clear-chat');
    const typingIndicator = document.querySelector('.typing-indicator');
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
    
    clearChatBtn.addEventListener('click', clearChat);
    marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: true,
        mangle: false
    });
    
    document.querySelectorAll('.bot-message:not(.typing-indicator)').forEach(el => {
        el.innerHTML = marked.parse(el.textContent);
    });

    chatContainer.scrollTop = chatContainer.scrollHeight;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = input.value.trim();
        if (!question) return;

        addMessage(question, 'user-message');
        input.value = '';
        submitBtn.disabled = true;
        typingIndicator.style.display = 'flex';
        chatContainer.scrollTop = chatContainer.scrollHeight;

        try {
            const response = await fetch('/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({ message: question })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            addMessage(data.answer, 'bot-message');

        } catch (error) {
            addMessage('Sorry, something went wrong. Please try again.', 'bot-message');
            console.error('Error:', error);
        } finally {
            typingIndicator.style.display = 'none';
            submitBtn.disabled = false;
            input.focus();
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    });

    function addMessage(text, className) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${className}`;
        
        if (className === 'bot-message') {
            messageDiv.innerHTML = marked.parse(text);
        } else {
            messageDiv.textContent = text;
        }
        
        chatContainer.insertBefore(messageDiv, typingIndicator);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return messageDiv;
    }

    async function clearChat() {
        if (!confirm('Are you sure you want to clear the chat and start a new conversation?')) {
            return;
        }

        try {
            chatContainer.innerHTML = '';
            chatContainer.appendChild(typingIndicator); // Re-add the typing indicator
            
            addMessage('Hello! How can I help you with information about SKIT today?', 'bot-message');
            
            await fetch('/clear-chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                }
            });
            
            input.focus();
        } catch (error) {
            console.error('Error clearing chat:', error);
            alert('There was an error clearing the chat. Please try again.');
        }
    }
});