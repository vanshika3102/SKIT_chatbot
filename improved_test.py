import google.generativeai as genai
import time
import os

# --- Configuration ---
# It's best practice to set your API key as an environment variable.
# You can get your key from Google AI Studio: https://aistudio.google.com/app/apikey
# In your terminal, you can run: export GEMINI_API_KEY="YOUR_API_KEY"
try:
    genai.configure(api_key="AIzaSyAH3nnFbRiyLdpC29KREfvKLq3QiOAP5zw")
except KeyError:
    print("ERROR: GEMINI_API_KEY environment variable not set.")
    print("Please set your API key to run this script.")
    exit()


# --- Model and Generation Settings ---
# These settings help control the output for better performance and accuracy.
generation_config = {
  "temperature": 0.2,       # Lower temperature for more factual, less creative answers.
  "top_p": 0.9,
  "top_k": 32,
  "max_output_tokens": 512, # Control max length to reduce total generation time.
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", # Using the fast and capable 1.5 Flash model
    generation_config=generation_config,
)

def ask_chatbot(question: str, source_url: str = None):
    """
    Asks the chatbot a question, with optional URL for context.
    Implements prompt engineering, URL grounding, and response streaming.
    """
    # 1. --- Prompt Engineering ---
    # A clear, instructive prompt that tells the model exactly how to behave.
    prompt_parts = [
        "You are a helpful and concise college assistant chatbot.",
        "Your primary goal is to answer questions based *only* on the context provided from the given URL.",
        "Do not use any external knowledge or make up information.",
        "If the answer is not found in the provided text, state that you could not find the answer in the document.",
        "Keep your answers brief and to the point.",
        f"\nUser Question: '{question}'",
    ]

    # 2. --- Grounding with Tools (URL Context) ---
    # The 'tools' parameter is the correct and most reliable way to make
    # the model fetch and use the content of a URL for its answer.
    tools = []
    if source_url:
        print(f"🧠 Grounding query in content from: {source_url}")
        try:
            # The model will fetch this URL and use its content to answer.
            # Note: This uses the File API to handle the URL.
            url_file = genai.upload_file(path=source_url, display_name="Source URL for Context")
            tools = [url_file]
        except Exception as e:
            print(f"\nCould not process the URL. Error: {e}")
            return # Exit if the URL is invalid or inaccessible
    else:
        # If no URL is provided, we can't ground the answer.
        prompt_parts.append("\nNote: No specific URL was provided to source the answer.")


    # 3. --- API Call with Streaming ---
    # Streaming the response greatly improves the user's perception of speed.
    print("\n🤖 Chatbot (streaming live response):")
    try:
        # Call the API with stream=True to get the response chunk by chunk.
        response_stream = model.generate_content(
            prompt_parts,
            tools=tools,
            stream=True
        )

        full_response = ""
        for chunk in response_stream:
            # Print each part of the response as it arrives.
            # The 'flush=True' ensures it appears on the screen immediately.
            print(chunk.text, end="", flush=True)
            full_response += chunk.text
        
        # Print a final newline for clean formatting after the stream completes.
        print()

    except Exception as e:
        print(f"\nAn error occurred during API call: {e}")

# --- --- --- --- ---
#      MAIN DEMO
# --- --- --- --- ---

if __name__ == "__main__":
    # --- Example 1: A general question without a specific source URL ---
    print("--- DEMO 1: General Question ---")
    start_time = time.time()
    ask_chatbot("What are some key features of the Gemini API?")
    end_time = time.time()
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")
    print("-" * 35)

    # --- Example 2: A course-specific question using a URL for grounding ---
    # We will use a public URL from the official documentation for this demo.
    # Replace this with the actual URL from your college's course page.
    print("\n--- DEMO 2: Grounded Question with URL ---")
    course_question = "give me the list of modules which are focused in skit training programe"
    # A real-world URL.
    course_url = "https://www.skit.ac.in/training-and-placement-soft-skills.html"
    
    start_time = time.time()
    ask_chatbot(course_question, source_url=course_url)
    end_time = time.time()
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")
    print("-" * 35)

    # --- Example 3: A question where the answer is not on the page ---
    print("\n--- DEMO 3: Question Not Answerable from URL ---")
    irrelevant_question = "What is the tuition fee for the Gemini API course?"
    
    start_time = time.time()
    ask_chatbot(irrelevant_question, source_url=course_url)
    end_time = time.time()
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")
    print("-" * 35)