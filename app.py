from utils.parser import parse_chat_log
from utils.message_reader import reader
from analysis.rag import TanglishChatRAG
from chat.assistant import ChatRAGAssistant


print("Loading chat data...")
text = reader(r"C:\Users\akhsh\Desktop\Fun Projects\Whatsapp-Process\chat.txt")
df = parse_chat_log(text)
print(f"Loaded {len(df)} messages")

# Initialize RAG
print("Setting up RAG system...")
chat_rag = TanglishChatRAG(df=df) 
print("RAG system ready!")

# Create bot with DeepSeek
assistant = ChatRAGAssistant(chat_rag, "deepseek-r1:8b")

# Example queries (programmatic usage)
print("\n=== Example Queries ===")

queries = [
    "Were there any fights in the chat?",
    "Is anyone angry for any reason?",
    "Find messages similar to 'good morning everyone'"
]

for query in queries:
    print(f"\nQ: {query}")
    print("Processing...")
    answer = assistant.query(query)
    print(f"A: {answer}")

while True:
    try:
        user_input = input("\nAsk me about your chat: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Get and display answer
        print("Thinking...")  # Visual feedback since DeepSeek might take time
        answer = assistant.query(user_input)
        print(f"\n{answer}")
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {str(e)}")
