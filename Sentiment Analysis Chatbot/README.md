# 💬 Sentiment Analysis Chatbot — BERT NLP

**Author:** Vaibhav Gupta | IIT Kanpur & IIT Delhi Certified  
**Stack:** Python · HuggingFace Transformers · BERT · Flask

---

## Overview
Customer support chatbot that:
- Classifies sentiment (Positive / Negative / Neutral) using **DistilBERT**
- Routes 60% of queries to automated responses
- Escalates negative-sentiment technical/refund queries to human agents
- Tracks satisfaction scores per session

## Architecture
```
User Message
      ↓
Sentiment Analysis (BERT → TextBlob → Rule-based fallback)
      ↓
Intent Classification (regex rules)
      ↓
Routing: Automate OR Escalate
      ↓
Response Generation + Session Tracking
```

## Quick Start
```bash
pip install -r requirements.txt
# Demo run (no GPU needed — CPU fallback)
python chatbot.py

# Start Flask API
from chatbot import create_app
app = create_app()
app.run(port=5001)
```

## API
```bash
# Start conversation
curl -X POST http://localhost:5001/chat \
  -d '{"message": "My app keeps crashing!"}'

# Get session summary
curl http://localhost:5001/session/<session_id>/summary
```

## Fallback Hierarchy
1. **BERT** (DistilBERT SST-2) — requires `torch` + `transformers`
2. **TextBlob** — lightweight, no GPU
3. **Rule-based** — always works, zero dependencies