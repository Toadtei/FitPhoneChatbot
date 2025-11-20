"""
FitPhone Chatbot - Conversational AI for Healthy Smartphone Habits

Knowledge Base JSON Structure (knowledge_base.json):
[
  {
    "q": "How can I reduce my screen time?",
    "a": "Start by tracking your daily usage and setting goals...",
    "source": "https://fitphone.nl/resources/screen-time",
    "category": "screen_time" 
  }
]
"""
#CATEGORY in the knowledgebase is something we can use later, it is not being used right now

import json
import re
import os
from datetime import datetime
from typing import List, Dict, Set
import threading
import queue

import tkinter as tk
from tkinter import scrolledtext, ttk

# pip install sentence-transformers
from sentence_transformers import SentenceTransformer, util
import numpy as np


class Config:
    """Configuration settings"""
    OLLAMA_MODEL = "phi3:mini"
    OLLAMA_ENDPOINT = "http://localhost:11434"
    KB_PATH = "knowledgebase.json"
    CONTEXT_WINDOW_SIZE = 8 # number of messages we are sending to the LLM to proccess and understand the context, 
    MAX_CONVERSATION_HISTORY = 10 #maybe for future, how many messages are we storing in genearl, can be later use for things like summarize the conversation
    LOG_FILE = "fitbot.log"
    

class Logger:
    """Simple logging utility that writes to file instead of console - code created with LLM help"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        # Clear log file on startup
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"FitBot Log Started: {datetime.now()} ===\n\n")
    
    def log(self, message: str, level: str = "INFO"):
        """Write a log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def error(self, message: str):
        """Log an error"""
        self.log(message, "ERROR")
    
    def info(self, message: str):
        """Log general info"""
        self.log(message, "INFO")
    
    def success(self, message: str):
        """Log success message"""
        self.log(message, "SUCCESS")


class KnowledgeBaseMatcher: 
    """
    Matches user queries with KB using sentence embeddings.
    This replaces the old Token Cache with an Embedding Cache.
    """
    
    def __init__(self, kb_path: str, logger: Logger):
        self.logger = logger
        self.kb = self._load_kb(kb_path)
        
        # 1. Load the Model (The heaviest part, happens once, takes some time
        self.logger.info("Loading sentence embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # BUILD THE CACHE
        # We pre-calculate the "meaning" of every KB entry now.
        # This is much faster than doing it every time a user asks a question.
        kb_texts = []
        for entry in self.kb:
            # We combine Question and Answer for better context matching
            kb_texts.append(f"{entry['q']} {entry['a']}")
        
        self.logger.info(f"Building Embedding Cache for {len(kb_texts)} entries...")
        
        # This variable 'self.kb_embeddings' IS the cache, streos vecotrs of KB in memory
        self.kb_embeddings = self.model.encode(
            kb_texts,
            normalize_embeddings=True,
            convert_to_tensor=True
        )
        self.logger.success("KB Embedding Cache ready")

    def _load_kb(self, path: str) -> List[Dict]:
        """Load knowledge base from JSON"""
        
        # checks if the jsnon file exists with specifide path from class Config
        if not os.path.exists(path):
            self.logger.error(f'Knowledge base not found: {path}')
            exit(1)

        # reads the json KB
        with open(path, 'r', encoding='utf-8') as f:
            kb = json.load(f)
        
        #checks if all Q&A pairs have q (question) and a (answer) parameters
        if not all("q" in e and "a" in e for e in kb):
            self.logger.error("KB validation failed: missing 'q' or 'a' fields")
            raise ValueError("Each KB entry must contain 'q' and 'a' fields.")
        
        self.logger.success(f"Loaded {len(kb)} KB entries")
        return kb
    
    def get_best_matches(self, user_input: str, top_k: int = 3) -> List[Dict]:
        """
        Find the KB entries with closest meaning to the user's message,
        using cosine similarity between embeddings.
        """
        user_input = user_input.strip()
        if not user_input:
            return []

        # Convert the user message into an embedding vector
        query_embedding = self.model.encode(
            user_input,
            normalize_embeddings=True,
            convert_to_tensor=True
        )

        # Compare user embedding with all KB embeddings
        # util.cos_sim returns a similarity score for each KB entry
        sims = util.cos_sim(query_embedding, self.kb_embeddings)[0].cpu().tolist()

        # Sort KB entries by highest similarity score
        ranked_indices = sorted(
            range(len(sims)),
            key=lambda i: sims[i],
            reverse=True
        )[:top_k]

        # Build the final results
        results: List[Dict] = []
        for idx in ranked_indices:
            entry = self.kb[idx]
            score = float(sims[idx])

            results.append({
                "question": entry["q"],
                "answer": entry["a"],
                "source": entry.get("source", ""),
                "category": entry.get("category", "general"),
                "score": score
            })

        return results


class OllamaClient:
    """Ollama LLM client with streaming""" # handles connection with our model running in Ollama and streaming back the output
    
    def __init__(self, model: str, endpoint: str, logger: Logger):
        self.model = model
        self.endpoint = endpoint
        self.logger = logger
        self._check_connection()
    
    def _check_connection(self):
        """Verify Ollama is running"""
        try:
            import requests
            response = requests.get(f"{self.endpoint}/api/tags", timeout=2)
            if response.status_code == 200:
                self.logger.success(f"Connected to Ollama: {self.model}")
            else:
                self.logger.error(f"Ollama returned status {response.status_code}")

        except Exception as e:
            self.logger.error(f"Cannot connect to Ollama: {e}")
    
    def generate_stream(self, messages: List[Dict], stream_callback=None):
        """Generate response with streaming output"""
        #UPDATED to send strucutred messages isnted of raw text
        import requests
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }
        full_response = ""
        
        try:
            response = requests.post(
                f"{self.endpoint}/api/chat", # prevously was geenrate by mistake
                json=payload,
                stream=True,
                timeout=60
            )
            
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        # The chat API returns content in 'message' -  'content'
                        if 'message' in chunk and 'content' in chunk['message']:
                            token = chunk['message']['content']
                            full_response += token
                            if stream_callback:
                                stream_callback(token)
                        
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
            
            return full_response
            
        except Exception as e:
            error_msg = f"Error connecting to AI: {str(e)}"
            self.logger.error(error_msg)
            if stream_callback:
                stream_callback(error_msg)
            return error_msg


class ConversationManager:
    """Manages conversation flow and context""" 
    # This class handles the context, KB retrieval, and constructing the message list for the Chat API
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.kb_matcher = KnowledgeBaseMatcher(config.KB_PATH, logger)
        self.ollama = OllamaClient(config.OLLAMA_MODEL, config.OLLAMA_ENDPOINT, logger)
        self.messages = [] 
        # Stores messages as list of simple dicts: {"role": "user", "content": "..."}
    
    def process_message(self, user_input: str, stream_callback=None) -> str:
        """Process user message and generate response"""
        
        # Get KB matches
        kb_matches = self.kb_matcher.get_best_matches(user_input, top_k=2)
        
        # Check matches and extract source
        match_source = None
        kb_context_str = ""
        
        # relevancy treshhold check
        if kb_matches and kb_matches[0]['score'] > 0.35:
            best_match = kb_matches[0]
            #  string to inject into the System Prompt
            kb_context_str = f"""
RELEVANT KNOWLEDGE BASE INFO:
Question: {best_match['question']}
Answer: {best_match['answer']}
"""
            if best_match.get('source'):
                match_source = best_match['source']

        api_messages = []
        
        # A: System Prompt With KB info injected
        base_system_prompt = self._get_system_prompt()
        if kb_context_str:
            full_system_content = base_system_prompt + kb_context_str
        else:
            full_system_content = base_system_prompt + "\nNo specific Knowledge Base info found for this query. Answer generally based on healthy digital habits."
            
        api_messages.append({"role": "system", "content": full_system_content})
        
        # Chat History (Last N messages)
        # We iterate through self.messages which are already in format {"role": "...", "content": "..."}
        recent_history = self.messages[-self.config.CONTEXT_WINDOW_SIZE:]
    
        # This leaves the 'datetime' object behind so it is not send to the ai model
        for msg in recent_history:
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
            
        #Current User Input
        api_messages.append({"role": "user", "content": user_input})

        # Send to Ollama (Chat Mode)
        response_text = self.ollama.generate_stream(api_messages, stream_callback)
        
        # Post-processing-Removes "FitBot:" if the model accidentally outputted it
        response_text = re.sub(r'^FitBot:\s*', '', response_text.strip(), flags=re.IGNORECASE)
        
        # 6. Adds Source
        if match_source and len(response_text) > 5:
            source_text = f"\n\nSource: {match_source}"
            response_text += source_text
            if stream_callback:
                stream_callback(source_text)
        
        # Save to history
        self._add_message("user", user_input)
        self._add_message("assistant", response_text)
        
        return response_text
    
    def _get_system_prompt(self) -> str:
        """System prompt defining bot personality""" 
        # UPDATED: for better Chat API performance
        return """You are FitBot, a down-to-earth but supportive AI assistant helping young adults with healthy smartphone habits.

CORE INSTRUCTIONS:
1. You are a supportive friend, not a robot.
3. Tone: Causal, direct, calm, and empathetic when required. 
2. Keep responses SHORT (2-4 sentences max). Do not lecture.
3. If RELEVANT KNOWLEDGE BASE INFO is provided below, use it as the primary source of truth.
4. If the user is just saying "hi", greeting you, or engages in normal conversation reply naturally without pushing advice, be a supportive and listeing friend.
5. If the user asks a question and NO KB INFO is provided, give general, safe advice about digital well-being, but disclose that this information is not in your knowledgebase yet.
5. SAFETY: Never diagnose medical conditions. If a user mentions self-harm or severe distress, suggest professional help immediately.

IMPORTANT:
- Do NOT start your response with "FitBot:".
- Do NOT repeat greetings if the conversation history shows we have already greeted.
- Speak naturally using "I" and "You"."""

    def _add_message(self, role: str, content: str):
        """Add message to history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
        
        # Removes the oldest message if max history is reached
        if len(self.messages) > self.config.MAX_CONVERSATION_HISTORY:
            self.messages.pop(0) # removes the first item

# COMPLETELY NEW: GUI Version of ChatInterface
class ChatInterface:
    """Tkinter GUI chat interface"""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.conversation = ConversationManager(config, logger)
        self.token_queue = queue.Queue()
        self.is_processing = False
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("FitBot - Healthy Smartphone Habits")
        
        # Launch in fullscreen/maximized
        self.root.state('zoomed')  # For Windows
        # Alternative for other platforms:
        # self.root.attributes('-zoomed', True)  # Linux
        # self.root.attributes('-fullscreen', True)  # macOS
        
        self.root.configure(bg="#1e1e1e")
        
        # Colors
        self.bg_color = "#1e1e1e"
        self.chat_bg = "#2d2d2d"
        self.user_bubble = "#3d5a80"
        self.bot_bubble = "#2d4356"
        self.text_color = "#e0e0e0"
        self.input_bg = "#3d3d3d"
        
        self._create_widgets()
        self._show_welcome_message()
        
        # Start token processing loop
        self._process_token_queue()
    
    def _create_widgets(self):
        """Create GUI elements"""
        
        # Header
        header = tk.Frame(self.root, bg="#d17842", height=60)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        
        title = tk.Label(header, text="🤖 FitBot - Healthy Smartphone Habits", 
                        font=("Arial", 20, "bold"),
                        bg="#d17842", fg="white")
        title.pack(pady=15)
        
        # Chat display area
        chat_frame = tk.Frame(self.root, bg=self.chat_bg)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Arial", 11),
            bg=self.chat_bg,
            fg=self.text_color,
            relief=tk.FLAT,
            state=tk.DISABLED,
            spacing1=2,
            spacing3=2
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags
        self.chat_display.tag_config("user", foreground="#7dd3fc", font=("Arial", 11, "bold"))
        self.chat_display.tag_config("bot", foreground="#fb923c", font=("Arial", 11, "bold"))
        self.chat_display.tag_config("thinking", foreground="#fb923c", font=("Arial", 11, "bold"))
        
        # Input area
        input_frame = tk.Frame(self.root, bg=self.bg_color)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)
        
        self.input_field = tk.Text(
            input_frame,
            height=3,
            font=("Arial", 11),
            bg=self.input_bg,
            fg=self.text_color,
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.input_field.bind("<Return>", self._on_enter)
        self.input_field.bind("<Shift-Return>", lambda e: None)  # Allow Shift+Enter for newline
        
        self.send_button = tk.Button(
            input_frame,
            text="Send",
            command=self._send_message,
            bg="#d17842",
            fg="white",
            font=("Arial", 12, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            width=10
        )
        self.send_button.pack(side=tk.RIGHT)
        
        self.input_field.focus()
    
    def _show_welcome_message(self):
        """Display welcome message"""
        welcome = """Welcome to FitBot! 🎉

I'm here to help you develop healthier smartphone habits!

Topics I can help with:
📱 Screen time • 😰 FOMO • 🔕 Notifications • 😴 Sleep
🧘 Digital detox • 🎯 Focus • 📊 Social media

What would you like to talk about?"""
        self._append_message("bot", welcome)
    
    def _append_message(self, role: str, message: str, tag=None):
        """Append message to chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        if role == "user":
            self.chat_display.insert(tk.END, "You: ", "user")
        elif role == "bot":
            self.chat_display.insert(tk.END, "FitBot: ", "bot")
        elif role == "thinking":
            self.chat_display.insert(tk.END, "FitBot: ", "thinking")
        
        if tag:
            self.chat_display.insert(tk.END, message + "\n\n", tag)
        else:
            self.chat_display.insert(tk.END, message + "\n\n")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def _append_text(self, text: str):
        """Append text without label (for streaming)"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, text)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def _on_enter(self, event):
        """Handle Enter key press"""
        if not event.state & 0x1:  # Check if Shift is not pressed
            self._send_message()
            return "break"  # Prevent newline
    
    def _send_message(self):
        """Send user message"""
        if self.is_processing:
            return
            
        user_input = self.input_field.get("1.0", tk.END).strip()
        
        if not user_input:
            return
        
        # Clear input field
        self.input_field.delete("1.0", tk.END)
        
        # Display user message
        self._append_message("user", user_input)
        
        # Disable input while processing
        self.is_processing = True
        self.send_button.config(state=tk.DISABLED)
        self.input_field.config(state=tk.DISABLED)
        
        # Show thinking indicator
        self._append_message("thinking", "🤔 Thinking...")
        
        # Process in background thread
        threading.Thread(target=self._process_response, args=(user_input,), daemon=True).start()
    
    def _process_response(self, user_input: str):
        """Process message in background thread"""
        
        # Callback to queue tokens
        def stream_callback(token):
            self.token_queue.put(('token', token))
        
        # Signal to remove thinking and start response
        self.token_queue.put(('start', None))
        
        # Generate response
        self.conversation.process_message(user_input, stream_callback)
        
        # Signal completion
        self.token_queue.put(('done', None))
    
    def _process_token_queue(self):
        """Process tokens from queue in main thread"""
        try:
            while True:
                msg_type, data = self.token_queue.get_nowait()
                
                if msg_type == 'start':
                    # Remove thinking indicator
                    self.chat_display.config(state=tk.NORMAL)
                    self.chat_display.delete("end-3l", "end-1l")
                    self.chat_display.insert(tk.END, "FitBot: ", "bot")
                    self.chat_display.config(state=tk.DISABLED)
                    
                elif msg_type == 'token':
                    self._append_text(data)
                    
                elif msg_type == 'done':
                    self._append_text("\n\n")
                    self.is_processing = False
                    self.send_button.config(state=tk.NORMAL)
                    self.input_field.config(state=tk.NORMAL)
                    self.input_field.focus()
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(10, self._process_token_queue)
    
    def start(self):
        """Start the GUI"""
        self.root.mainloop()




def main():
    """Main entry point"""
    config = Config()
    logger = Logger(config.LOG_FILE)
    logger.info("Starting FitBot")
    chat = ChatInterface(config, logger)
    chat.start()


if __name__ == "__main__":
    main()

# OLD TERMINAL BASED CHAT INTERFACE

# class ChatInterface:
#     """Terminal chat interface""" # the UI interface, a terminal for now, triggered by the launched main class, ChatInterface launches its new ConversationManager
    
#     def __init__(self, config: Config):
#         self.config = config
#         self.conversation = ConversationManager(config)
    
#     def start(self):
#         """Start chat session"""
#         self._print_welcome()
        
#         while True:
#             try:
#                 user_input = input("\n\033[1;36mYou:\033[0m ").strip()
                
#                 if not user_input:
#                     continue
                
#                 # a quick way how users can exit the app for now since we are using only cmd interface
#                 if user_input.lower() in ["quit", "exit", "bye"]:
#                     self._print_goodbye()
#                     break
#                 # sends the users input, the message, to the ConversationManager for proccessing
#                 self.conversation.process_message(user_input)
                
#             except KeyboardInterrupt:
#                 print("\n")
#                 self._print_goodbye()
#                 break
#             except Exception as e:
#                 print(f"\nError: {e}")
    
#     def _print_welcome(self):
#         """Print welcome message"""
#         print(""" \033[38;5;208m  
#          ______          ______             
#         (______)(_)  _  (____  \\        _   
#          _____   _ _| |_ ____)  ) ___ _| |_ 
#         |  ___) | (_   _)  __  ( / _ (_   _)
#         | |     | | | |_| |__)  ) |_| || |_ 
#         |_|     |_|  \\__)______/ \\___/  \\__)
    
            
# ╔═══════════════════════════════════════════════════════╗
# ║                 Welcome to FitBot!                    ║
# ║      Your AI Assistant for Healthy Smartphone Use     ║
# ╚═══════════════════════════════════════════════════════╝
# \033[0m

# I'm here to help you develop healthier smartphone habits!

# Topics I can help with:
# 📱 Screen time • 😰 FOMO • 🔕 Notifications • 😴 Sleep
# 🧘 Digital detox • 🎯 Focus • 📊 Social media

# Type 'quit' to exit.
# """)
    
#     def _print_goodbye(self):
#         """Print goodbye message"""
#         print("\n" + "="*60)
#         print("Thank you for chatting! Take care!")
#         print("Remember: You're in control of your digital wellbeing.")
#         print("="*60 + "\n")



# main entry point for this python app

