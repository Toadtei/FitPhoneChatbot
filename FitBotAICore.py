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
import sys, time, threading



class Config:
    """Configuration settings"""
    OLLAMA_MODEL = "phi3:mini"
    OLLAMA_ENDPOINT = "http://localhost:11434"
    KB_PATH = "knowledgebase.json"
    CONTEXT_WINDOW_SIZE = 8 # number of messages we are sending to the LLM to proccess and understand the context, 
    MAX_CONVERSATION_HISTORY = 10 #maybe for future, how many messages are we storing in genearl, can be later use for things like summarize the conversation

class KnowledgeBaseMatcher: # basically very simple version of RAG retrievel, instead of documents and vecotr database, we have Q&A pairs in json. 
    # TODO: move from jaccard similarity matching to sentecne embedings and vectorizing to match better based on meaning of the input.
    """Matches user queries with knowledge base""" 
    
    
    def __init__(self, kb_path: str):
        self.kb = self._load_kb(kb_path)
        self.stopwords = {"a", "an", "the", "is", "are", "was", "were",
                         "in", "on", "at", "to", "for", "of", "and", "or"}
    
    def _load_kb(self, path: str) -> List[Dict]:
        """Load knowledge base from JSON"""
        
        # checks if the jsnon file exists with specifide path from class Config
        if not os.path.exists(path):
            print(f"\nERROR: Knowledge base not found: {path}")
            exit(1)

        # reads the json KB
        with open(path, 'r', encoding='utf-8') as f:
            kb = json.load(f)
        
        #checks if all Q&A pairs have q (question) and a (answer) parameters
        if not all("q" in e and "a" in e for e in kb):
            raise ValueError("Each KB entry must contain 'q' and 'a' fields.")
        
        print(f"✓ Loaded {len(kb)} KB entries")
        return kb
    
    def get_best_matches(self, user_input: str, top_k: int = 3) -> List[Dict]:
        """Get top K most relevant KB entries"""
        
        user_tokens = self._tokenize(user_input.lower())
        
        scores = []
        for entry in self.kb:
            question_tokens = self._tokenize(entry['q'].lower())
            score = self._jaccard_similarity(user_tokens, question_tokens)
            
            scores.append({
                "question": entry['q'],
                "answer": entry['a'],
                "source": entry.get('source', ''),
                "category": entry.get('category', 'general'),
                "score": score
            })
        
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]
    
    def _tokenize(self, text: str) -> Set[str]:
        """Extract meaningful words"""
        words = re.findall(r'\b\w+\b', text)
        return {w for w in words if len(w) >= 3 and w not in self.stopwords}
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate similarity score"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


class OllamaClient:
    """Ollama LLM client with streaming""" # handles connection with our model running in Ollama and streaming back the output
    
    def __init__(self, model: str, endpoint: str):
        self.model = model
        self.endpoint = endpoint
        self._check_connection()
    
    def _check_connection(self):
        """Verify Ollama is running"""

        #TODO: for production/demo, we should change printing out errors to the UI chat, lets create a log file and print it there
        try:
            import requests
            response = requests.get(f"{self.endpoint}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✓ Connected to Ollama: {self.model}")
            else:
                print(f"Ollama error (status {response.status_code})")
                exit(1)
        except Exception as e:
            print(f"Cannot connect to Ollama: {e}")
            exit(1)
    
    def generate_stream(self, prompt: str, system_prompt: str = ""):
        """Generate response with streaming output"""
        import requests
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True
        }
        
        try:
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                stream=True, # doenst wait for the whole answer and then displys it, but streams it back in chunks, feature of Ollamas APIs
                timeout=60 # time in seconds till python waits for the models response
            )
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            print(token, end="", flush=True)
                            full_response += token
                        
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
            
            print()  # New line after streaming
            return full_response
            
        except Exception as e:
            print(f"\n Generation error: {e}") #TODO: log in log file during demo/production
            return "I'm having trouble generating a response. Please try again."


class ConversationManager:
    """Manages conversation flow and context""" # This class handles the context of the conversation, keeping track of past messages, the overall mamangerm and uses KowledgeBaseMatcher to find best knowledge and then OllamaClient class to connect to the model.
    
    def __init__(self, config: Config):
        self.config = config
        self.kb_matcher = KnowledgeBaseMatcher(config.KB_PATH)
        self.ollama = OllamaClient(config.OLLAMA_MODEL, config.OLLAMA_ENDPOINT)
        self.messages = []
    
    def process_message(self, user_input: str) -> str:
        """Process user message and generate response"""
        self._add_message("user", user_input)
        
        # Get KB matches
        kb_matches = self.kb_matcher.get_best_matches(user_input, top_k=3)
        
        # Build prompt
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_prompt(user_input, kb_matches)
        
        
        # Print FitBot: and stay on the same line
        print("\033[38;5;208mFitBot:\033[0m ", end="", flush=True)

        # prints thinking message and deletes after 2 seconds TODO: print thinking and delete it right before the model strats streaming, not fixed seconds like it is right now.
        thinking_text = "🤔 Thinking..."
        print(thinking_text, end="", flush=True)

        time.sleep(2)
        sys.stdout.write("\b" * len(thinking_text))
        sys.stdout.write(" " * len(thinking_text))
        sys.stdout.write("\b" * (len(thinking_text)+1))
        sys.stdout.flush()
        

        # Generate response (streaming)hh
        response = self.ollama.generate_stream(user_prompt, system_prompt)

        # clenas up prefix FitBot: that smaller models Phi3:mini add
        response = re.sub(r'^FitBot:\s*', '', response.strip(), flags=re.IGNORECASE)

        
        # Add source if available
        if kb_matches and kb_matches[0]['score'] > 0 and kb_matches[0].get('source'):
            source_text = f"\nSource: {kb_matches[0]['source']}"
            print(source_text)
            response += source_text
        
        self._add_message("assistant", response)
        return response
    
    def _get_system_prompt(self) -> str:
        """System prompt defining bot personality""" # the base prompt, this prompt defines how our ai model should act, SYSTEM PROMPT, 
        #TODO: prompt engineering, experiments and research how to imporve and get better results
        return """You are FitBot, a friendly AI assistant helping young adults with healthy smartphone habits.

PERSONALITY:
- Warm, conversational, supportive friend
- Empathetic and non-judgmental
- Casual natural language, not preachy
- Personal and relatable

KNOWLEDGE BASE USAGE:
- Use KB information as foundation for factual claims
- Do not halucinate
- For greetings/chitchat, respond naturally without forcing KB info
- Combine KB entries naturally when relevant
- If user asks something not in KB, acknowledge and inform about that but offer related topics

RESPONSE STYLE:
- Address their SPECIFIC situation
- Keep responses SHORT and conversational, 1-3 sentences typically, max 6,
- Build on conversation naturally respecting the previous messages, do NOT repeat greetings
- Ask follow-up questions when appropriate
- Be encouraging and realistic

BOUNDARIES:
- Never diagnose health conditions
- Provide info and support, not therapy
- Suggest professional help when needed, especially for mentiones of self-harm, suicide, or medical emergency
- If the conversation flows off-topic, gently redirect
    
IMPORTANT:
- DO NOT start your response with "FitBot:" - just respond directly
- DO NOT repeat greetings if you've already greeted them
- Keep track of what's been discussed and build on it"""

    def _build_prompt(self, user_input: str, kb_matches: List[Dict]) -> str:
        """Build complete prompt with context and KB""" # builds the whole promtp that is being send to the ai model, indluces the previous system prompt, user message, contex(the message history up until now), relevant matched KB Q&A pairs and 
        # Format conversation history
        context = self._format_context()
        
        # Format KB matches
        kb_info = ""
        if kb_matches and any(m['score'] > 0 for m in kb_matches):
            kb_info = "\n\nRELEVANT KNOWLEDGE BASE:\n"
            for i, match in enumerate(kb_matches[:3], 1):
                if match['score'] > 0:
                    kb_info += f"{i}. Q: {match['question']}\n"
                    kb_info += f"   A: {match['answer']}\n"
                    kb_info += f"   Relevance: {match['score']:.2f}\n\n"
        
        return f"""CONVERSATION HISTORY:
{context}

CURRENT USER MESSAGE: "{user_input}"
{kb_info}

Instructions: 
- This is a CONTINUING conversation
- Keep your response SHORT (1-3 sentences)
- Build naturally on what was said before
- DO NOT start with "FitBot:" - respond directly
- Be conversational and natural

Your response:"""
    
    def _format_context(self) -> str:
        """Format recent conversation for context"""
        if len(self.messages) == 0:
            return "(Start of conversation)"
        
        # Get last N messages based on config
        recent = self.messages[-self.config.CONTEXT_WINDOW_SIZE:]
        
        if len(recent) == 0:
            return "(Start of conversation)"
        
        formatted = ""
        for msg in recent:
            role = "User" if msg["role"] == "user" else "FitBot"
            formatted += f"{role}: {msg['content']}\n"
        
        return formatted.strip()
    
    def _add_message(self, role: str, content: str):
        """Add message to history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
        
        # removes the oldest massege if the valuse of max conversation hisotry form the Config class is reached
        if len(self.messages) > self.config.MAX_CONVERSATION_HISTORY:
            self.messages = self.messages[-self.config.MAX_CONVERSATION_HISTORY:]


class ChatInterface:
    """Terminal chat interface""" # the UI interface, a terminal for now, triggered by the launched main class, ChatInterface launches its new ConversationManager
    
    def __init__(self, config: Config):
        self.config = config
        self.conversation = ConversationManager(config)
    
    def start(self):
        """Start chat session"""
        self._print_welcome()
        
        while True:
            try:
                user_input = input("\n\033[1;36mYou:\033[0m ").strip()
                
                if not user_input:
                    continue
                
                # a quick way how users can exit the app for now since we are using only cmd interface
                if user_input.lower() in ["quit", "exit", "bye"]:
                    self._print_goodbye()
                    break
                # sends the users input, the message, to the ConversationManager for proccessing
                self.conversation.process_message(user_input)
                
            except KeyboardInterrupt:
                print("\n")
                self._print_goodbye()
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    def _print_welcome(self):
        """Print welcome message"""
        print(""" \033[38;5;208m  
         ______          ______             
        (______)(_)  _  (____  \\        _   
         _____   _ _| |_ ____)  ) ___ _| |_ 
        |  ___) | (_   _)  __  ( / _ (_   _)
        | |     | | | |_| |__)  ) |_| || |_ 
        |_|     |_|  \\__)______/ \\___/  \\__)
    
            
╔═══════════════════════════════════════════════════════╗
║                 Welcome to FitBot!                    ║
║      Your AI Assistant for Healthy Smartphone Use     ║
╚═══════════════════════════════════════════════════════╝
\033[0m

I'm here to help you develop healthier smartphone habits!

Topics I can help with:
📱 Screen time • 😰 FOMO • 🔕 Notifications • 😴 Sleep
🧘 Digital detox • 🎯 Focus • 📊 Social media

Type 'quit' to exit.
""")
    
    def _print_goodbye(self):
        """Print goodbye message"""
        print("\n" + "="*60)
        print("Thank you for chatting! Take care!")
        print("Remember: You're in control of your digital wellbeing.")
        print("="*60 + "\n")

# main entry point for this python app
def main():
    """Main entry point"""
    print(" Starting...")
    config = Config()
    chat = ChatInterface(config)
    chat.start()


if __name__ == "__main__":
    main()