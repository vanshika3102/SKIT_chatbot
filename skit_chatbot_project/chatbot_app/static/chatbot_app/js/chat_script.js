// static/js/chat_script.js

document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.getElementById('chat-container');
    const form = document.getElementById('input-form');
    const input = document.getElementById('question-input');
    const submitBtn = document.getElementById('submit-btn');
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;

    // Scroll to the bottom of the chat on page load
    chatContainer.scrollTop = chatContainer.scrollHeight;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = input.value.trim();
        if (!question) return;

        // Display user's message immediately
        addMessage(question, 'user-message');
        input.value = '';
        submitBtn.disabled = true;

        // Create a placeholder for the bot's response
        const thinkingMessage = addMessage('...', 'bot-message');

        try {
            // The fetch URL should match your urls.py
            const response = await fetch('/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                // We only need to send the new message. The server knows the history.
                body: JSON.stringify({ message: question })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            // Update the placeholder with the actual answer
            thinkingMessage.textContent = data.answer;

        } catch (error) {
            thinkingMessage.textContent = 'Sorry, something went wrong. Please try again.';
            console.error('Error:', error);
        } finally {
            submitBtn.disabled = false;
            input.focus();
            chatContainer.scrollTop = chatContainer.scrollHeight; // Scroll to bottom
        }
    });

    function addMessage(text, className) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', className);
        messageDiv.textContent = text;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return messageDiv;
    }
});