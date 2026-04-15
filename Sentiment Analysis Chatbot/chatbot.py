"""
=============================================================
Sentiment Analysis Chatbot — BERT-based NLP
Author  : Vaibhav Gupta
Tech    : Python · Transformers (HuggingFace) · BERT · Flask
=============================================================
Implements a customer support chatbot that:
  1. Classifies sentiment of incoming messages (Positive / Negative / Neutral)
  2. Routes queries to automated responses (60% automation rate)
  3. Escalates negative-sentiment messages to human agents
  4. Tracks satisfaction scores per session
=============================================================
"""

import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Optional heavy imports ────────────────────────────────
try:
    import torch
    from transformers import pipeline as hf_pipeline
    HF_AVAILABLE = True
    logger.info("HuggingFace Transformers available — using BERT sentiment")
except ImportError:
    HF_AVAILABLE = False
    logger.warning("transformers/torch not installed — using rule-based fallback")

try:
    from textblob import TextBlob
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False


# ══════════════════════════════════════════════════════════
# 1. SENTIMENT ANALYSER
# ══════════════════════════════════════════════════════════
class SentimentAnalyser:
    """
    Three-tier strategy:
      1. BERT (distilbert-base-uncased-finetuned-sst-2-english) — most accurate
      2. TextBlob                                                — lightweight
      3. Rule-based keyword matching                            — always available
    """

    POSITIVE_WORDS = {
        "great", "excellent", "amazing", "wonderful", "fantastic", "good",
        "happy", "love", "like", "thank", "thanks", "helpful", "perfect",
        "awesome", "brilliant", "superb", "satisfied", "pleased",
    }
    NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "horrible", "hate", "dislike", "angry",
        "frustrated", "disappointed", "broken", "error", "fail", "failed",
        "problem", "issue", "wrong", "not working", "worst", "useless",
        "stupid", "crash", "bug", "refund",
    }

    def __init__(self):
        self.bert_pipe = None
        if HF_AVAILABLE:
            try:
                logger.info("Loading BERT sentiment model …")
                self.bert_pipe = hf_pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    return_all_scores=True,
                )
                logger.info("BERT model loaded ✓")
            except Exception as e:
                logger.warning("BERT load failed: %s. Using fallback.", e)

    def analyse(self, text: str) -> dict:
        text_lower = text.lower().strip()

        # Strategy 1: BERT
        if self.bert_pipe:
            return self._bert_sentiment(text, text_lower)

        # Strategy 2: TextBlob
        if TB_AVAILABLE:
            return self._textblob_sentiment(text, text_lower)

        # Strategy 3: Rule-based
        return self._rule_based(text_lower)

    def _bert_sentiment(self, text: str, text_lower: str) -> dict:
        scores = self.bert_pipe(text[:512])[0]  # BERT max 512 tokens
        score_map = {s["label"]: s["score"] for s in scores}
        pos = score_map.get("POSITIVE", 0)
        neg = score_map.get("NEGATIVE", 0)
        if pos > 0.65:
            sentiment, confidence = "positive", pos
        elif neg > 0.65:
            sentiment, confidence = "negative", neg
        else:
            sentiment, confidence = "neutral", max(pos, neg)
        return {"sentiment": sentiment, "confidence": round(confidence, 4), "engine": "bert"}

    def _textblob_sentiment(self, text: str, text_lower: str) -> dict:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            sentiment, confidence = "positive", min(0.5 + polarity * 0.5, 1.0)
        elif polarity < -0.1:
            sentiment, confidence = "negative", min(0.5 + abs(polarity) * 0.5, 1.0)
        else:
            sentiment, confidence = "neutral", 0.5
        return {"sentiment": sentiment, "confidence": round(confidence, 4), "engine": "textblob"}

    def _rule_based(self, text_lower: str) -> dict:
        words = set(re.findall(r'\b\w+\b', text_lower))
        pos_hits = len(words & self.POSITIVE_WORDS)
        neg_hits = len(words & self.NEGATIVE_WORDS)
        if neg_hits > pos_hits:
            return {"sentiment": "negative", "confidence": 0.70, "engine": "rule-based"}
        elif pos_hits > neg_hits:
            return {"sentiment": "positive", "confidence": 0.70, "engine": "rule-based"}
        return {"sentiment": "neutral", "confidence": 0.50, "engine": "rule-based"}


# ══════════════════════════════════════════════════════════
# 2. INTENT CLASSIFIER  (rule-based)
# ══════════════════════════════════════════════════════════
INTENT_RULES = {
    "refund"       : r"\b(refund|money back|return|cancel)\b",
    "billing"      : r"\b(bill|invoice|charge|payment|subscription)\b",
    "technical"    : r"\b(error|crash|bug|not working|broken|issue|problem|fail)\b",
    "account"      : r"\b(login|password|account|sign in|access|locked)\b",
    "shipping"     : r"\b(delivery|shipping|track|order|package|arrived)\b",
    "product_info" : r"\b(how|what|feature|plan|price|cost|tell me)\b",
    "greeting"     : r"\b(hi|hello|hey|good morning|good evening)\b",
    "farewell"     : r"\b(bye|goodbye|thank you|thanks|great|done)\b",
}

def classify_intent(text: str) -> str:
    t = text.lower()
    for intent, pattern in INTENT_RULES.items():
        if re.search(pattern, t):
            return intent
    return "general"


# ══════════════════════════════════════════════════════════
# 3. RESPONSE GENERATOR
# ══════════════════════════════════════════════════════════
AUTOMATED_RESPONSES = {
    "greeting"     : "Hello! 👋 Welcome to support. How can I help you today?",
    "farewell"     : "You're welcome! Have a great day! 😊 Is there anything else I can help with?",
    "billing"      : "I can see your billing details. Your current plan renews on the 1st of every month. For specific charge queries, please share your account ID.",
    "shipping"     : "You can track your order at our portal. Standard delivery takes 3–5 business days. Express shipping takes 1–2 days.",
    "product_info" : "I'd be happy to explain our plans! We offer Basic (₹499/mo), Standard (₹999/mo), and Premium (₹1,999/mo). Which features interest you?",
    "account"      : "For account issues, please click 'Forgot Password' on the login page. If locked out, I can escalate to our account team immediately.",
    "general"      : "Thank you for reaching out! Could you please provide more details so I can assist you better?",
}

ESCALATION_TRIGGERS = {"refund", "technical"}

def generate_response(intent: str, sentiment: str, text: str) -> dict:
    auto_handle   = intent not in ESCALATION_TRIGGERS
    needs_human   = sentiment == "negative" and intent in ESCALATION_TRIGGERS

    if needs_human:
        response   = ("I'm truly sorry to hear you're experiencing this issue. 😔 "
                      "I'm escalating your query to a senior agent right now. "
                      "You'll receive a callback within 30 minutes.")
        escalated  = True
        automated  = False
    elif auto_handle:
        response   = AUTOMATED_RESPONSES.get(intent, AUTOMATED_RESPONSES["general"])
        escalated  = False
        automated  = True
    else:
        response   = ("I understand your concern. Let me check this for you and "
                      "connect you with the right team.")
        escalated  = True
        automated  = False

    return {
        "response"  : response,
        "escalated" : escalated,
        "automated" : automated,
    }


# ══════════════════════════════════════════════════════════
# 4. SESSION MANAGER
# ══════════════════════════════════════════════════════════
class ChatSession:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id  = session_id or str(uuid.uuid4())[:8]
        self.turns       = []
        self.created_at  = datetime.utcnow()
        self.analyser    = SentimentAnalyser()

    def chat(self, user_message: str) -> dict:
        sentiment_result = self.analyser.analyse(user_message)
        intent           = classify_intent(user_message)
        response_data    = generate_response(intent, sentiment_result["sentiment"], user_message)

        turn = {
            "turn"       : len(self.turns) + 1,
            "timestamp"  : datetime.utcnow().isoformat(),
            "user"       : user_message,
            "sentiment"  : sentiment_result,
            "intent"     : intent,
            "bot"        : response_data["response"],
            "automated"  : response_data["automated"],
            "escalated"  : response_data["escalated"],
        }
        self.turns.append(turn)
        return turn

    def summary(self) -> dict:
        if not self.turns:
            return {}
        automated = sum(1 for t in self.turns if t["automated"])
        sentiments = [t["sentiment"]["sentiment"] for t in self.turns]
        return {
            "session_id"       : self.session_id,
            "total_turns"      : len(self.turns),
            "automated_turns"  : automated,
            "automation_rate"  : f"{automated / len(self.turns) * 100:.1f}%",
            "escalated"        : any(t["escalated"] for t in self.turns),
            "sentiment_summary": {
                "positive" : sentiments.count("positive"),
                "neutral"  : sentiments.count("neutral"),
                "negative" : sentiments.count("negative"),
            },
        }


# ══════════════════════════════════════════════════════════
# 5. FLASK API
# ══════════════════════════════════════════════════════════
def create_app():
    from flask import Flask, request, jsonify

    app      = Flask(__name__)
    sessions = {}   # In-production: use Redis

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "bert_available": HF_AVAILABLE})

    @app.route("/chat", methods=["POST"])
    def chat():
        data    = request.get_json(force=True)
        message = data.get("message", "").strip()
        sid     = data.get("session_id")

        if not message:
            return jsonify({"error": "message is required"}), 400

        if sid and sid in sessions:
            session = sessions[sid]
        else:
            session = ChatSession(session_id=sid)
            sessions[session.session_id] = session

        turn = session.chat(message)
        return jsonify({
            "session_id" : session.session_id,
            "bot_reply"  : turn["bot"],
            "sentiment"  : turn["sentiment"],
            "intent"     : turn["intent"],
            "escalated"  : turn["escalated"],
            "automated"  : turn["automated"],
        })

    @app.route("/session/<sid>/summary")
    def session_summary(sid):
        if sid not in sessions:
            return jsonify({"error": "Session not found"}), 404
        return jsonify(sessions[sid].summary())

    return app


# ══════════════════════════════════════════════════════════
# 6. DEMO
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  SENTIMENT ANALYSIS CHATBOT — Demo")
    print("=" * 60)

    session = ChatSession()
    test_messages = [
        "Hello! I need some help.",
        "What plans do you offer?",
        "My app keeps crashing and I'm really frustrated!",
        "I want a refund, this product is terrible.",
        "Thanks, I think I understand now. Bye!",
    ]

    for msg in test_messages:
        print(f"\n👤 User: {msg}")
        turn = session.chat(msg)
        print(f"🤖 Bot : {turn['bot']}")
        print(f"   Sentiment: {turn['sentiment']['sentiment']} "
              f"({turn['sentiment']['confidence']:.2f}) "
              f"[{turn['sentiment']['engine']}] | Intent: {turn['intent']} | "
              f"Automated: {turn['automated']} | Escalated: {turn['escalated']}")

    summary = session.summary()
    print(f"\n{'='*60}")
    print(f"📊 Session Summary:")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    print(f"\n✅ Chatbot demo complete!")