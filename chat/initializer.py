from assistant import ChatRAGAssistant

class WhatsAppChatBot:
    def __init__(self, chat_rag, model_name="deepseek-r1:8b"):
        self.assistant = ChatRAGAssistant(chat_rag, model_name)
    
    def ask(self, question: str) -> str:
        """Ask a question about the WhatsApp chat"""
        return self.assistant.query(question)
    
    def interactive_mode(self):
        """Run interactive chat mode"""
        print("WhatsApp Chat Assistant Ready! (type 'quit' to exit)")
        print("Using DeepSeek with <think> tag handling")
        print("-" * 50)
        
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
                answer = self.ask(user_input)
                print(f"\n{answer}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
