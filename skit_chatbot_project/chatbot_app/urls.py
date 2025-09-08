# chatbot_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # URL for the main chat page
    path('', views.chat_view, name='chat_page'),
    # URL for the API endpoint that answers questions
    path('chat/', views.chat_view, name='chat'),
    # URL for clearing chat history
    path('clear-chat/', views.clear_chat, name='clear_chat'),
]