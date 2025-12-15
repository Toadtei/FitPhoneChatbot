Structure:
   1. Configuration
   2. Utilities 
       #PromptInjectionDetector
       #InputSanitizer
   3. Core Logic
       #Knowledge Base Handler
       #Ollama Connection Client
       #Safety Filter
       #Prompt Builder
       #Conversation Context Manager
       #Core Message Processor
   4. GUI (UIConfig, UIMessages, Chat Interface)
   5. Main Entry Point

Rough flow:
1. GUI - User Input
2. Core Message Processor (Processes the message)
   3. InputSanitizer(length and some text type)
   4. PromptInjectionDetector
   5. InputSanitizer(removes confusing charachters to clean to prompt send to LLM)
   6. Safety Filter (checks for boundaries of users request)
   7. Knowledge Base Handler (finds best matching Q-A pairs)
   8. Conversation Context Manager (retrieves the conversation history)
   9. Prompt Builder (Builds the prompt send to LLM)
   10. Ollama Connection Client
   11. Safety Filter (checks the streamed output for boundaries violation)
12. GUI - message displayed


BEFORE YOU START WORKING:
- ALWAYS GIT PULL
- AFTER YOU FINISH, COMMIT AND PUSH
- COMMIT AND PUSH OFTEN
- DO NOT PUSH ERRORS
- WORK ONLY ON DEV BRANCH OR CREATE YOUR OWN FOR EXPERIMENTING, NEVER PUSH DIRECTLY TO MAIN

# FitBot - AI Chatbot for Healthy Smartphone Habits

A conversational AI assistant helping young adults develop healthier smartphone habits through friendly, empathetic conversations.

## Quick Start

### Prerequisites
1. **Python 3.8+** installed
2. **Ollama** installed and running locally

### Setup Steps

1. **Install Ollama**
```bash
   # Download from: https://ollama.com/download
   # Or use package manager (macOS/Linux)
   curl -fsSL https://ollama.com/install.sh | sh
```

2. **Download Mistral Model or Phi for better local testing**
```bash
   ollama pull phi3:mini
```

3. **Start Ollama Service**
```bash
   ollama serve
```
   Keep this terminal running in the background.

4. **Install Python Dependencies**
```bash
   pip install requests
```

5. **Run FitBot**
```bash
   python FitBotAICore.py
```

## Configuration

Edit the `Config` class in `FitBotAICore.py`:
- `OLLAMA_MODEL`: AI model to use (default: "mistral")
- `KB_PATH`: Path to knowledge base JSON file
- `MAX_CONVERSATION_HISTORY`: Number of messages to keep in context

## Architecture
```
User Input → KB Matcher (Jaccard Similarity) → Prompt Builder → Ollama LLM → Streamed Response
```

## Project Structure

- `FitBotAICore.py` - Main chatbot application
- `knowledge_base.json` - Q&A pairs database
- Terminal-based interface (GUI coming)

## Troubleshooting

**"Cannot connect to Ollama"**
- Ensure Ollama is running: `ollama serve`
- Check endpoint: http://localhost:11434

**"Knowledge base not found"**
- Create `knowledge_base.json` in the same directory as the script

**Model not responding**
- Verify Mistral is downloaded: `ollama list`
- Re-download if needed: `ollama pull mistral`

---

**FitPhone Project** | Fontys AI For Society Minor | Offline-first, privacy-focused AI assistant
