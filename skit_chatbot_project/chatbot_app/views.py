# chatbot_app/views.py

import json
from django.http import JsonResponse
from django.shortcuts import render
from . import chatbot_logic # Import your logic file

def chat_view(request):
    # This single view handles both rendering the page and the API calls

    # --- Handle POST requests (when user sends a message) ---
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # Note: The JS sends 'message', not 'question' now
            print(f"Received POST data: {data}")
            user_question = data.get('message')
            print(f"User question: {user_question}")
            # Get history from the session
            history = request.session.get('chat_history', [])

            if not user_question:
                return JsonResponse({'error': 'No message provided'}, status=400)

            # 1. Rewrite the query for better search
            standalone_question = chatbot_logic.rewrite_query_with_history(user_question, history)
            
            # 2. Search the index with the rewritten query
            context = chatbot_logic.search_combined_index(standalone_question)
            # print(f"context:{context}")
            # print(f"qqqqqqqqqqqqqqqq:{context}")
            # 3. Get the final answer from Gemini, providing the original question and history
            answer = chatbot_logic.ask_gemini(context, user_question, history)

            # 4. Update the session history
            history.append({"role": "user", "parts": [user_question]})
            history.append({"role": "model", "parts": [answer]})
            request.session['chat_history'] = history

            return JsonResponse({'answer': answer})
        except Exception as e:
            print(f"Error in chat view POST: {e}")
            return JsonResponse({'error': 'An internal error occurred.'}, status=500)

    # --- Handle GET requests (when user loads or refreshes the page) ---
    else:
        # Get the history from the session to display it on the page
        chat_history = request.session.get('chat_history', [])
        # Optional: Clear history for a new session if you want
        # request.session['chat_history'] = [] 
        
        return render(request, 'chatbot_app/chat.html', {
            'chat_history': chat_history
        })

def clear_chat(request):
    """View to clear the chat history."""
    if request.method == 'POST':
        # Clear the chat history from the session
        request.session['chat_history'] = []
        return JsonResponse({'status': 'success', 'message': 'Chat history cleared'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)