"""
Pragmatic Processing for Cybersecurity (Example)
Focus: "meaning in context" for SOC / phishing / threat intel text

What this script demonstrates:
1) Sentiment & "Urgency/Pressure" detection in phishing/social engineering text
2) Stance detection (Pro/Anti/Neutral) toward a target claim (e.g., "This exploit works")
3) Intent classification for SOC tickets (triage/routing) using zero-shot classification
4) Conversation context: resolve follow-up messages by inheriting context from previous turns

Install:
  pip install -U transformers torch sentencepiece scikit-learn

Notes:
- This is a safe, defensive educational demo (no exploit code).
- For Thai: choose multilingual models (e.g., xlm-roberta) or Thai-specific classifiers if available.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# -----------------------------
# 0) Sample cybersecurity texts
# -----------------------------
PHISHING_EMAILS = [
    {
        "id": "phish_01",
        "text": "URGENT: Your account will be locked in 2 hours. Verify your identity immediately: https://secure-login.example.com/reset",
        "label_hint": "phish",
    },
    {
        "id": "legit_01",
        "text": "Reminder: Your monthly statement is available in the portal. If you have questions, contact support.",
        "label_hint": "legit",
    },
    {
        "id": "phish_02",
        "text": "Action required now! We detected unusual sign-in. Confirm your password to keep access.",
        "label_hint": "phish",
    },
]

THREAT_FORUM_POSTS = [
    {
        "id": "forum_01",
        "target_claim": "This exploit works reliably on the latest version.",
        "text": "Tested it last night. Works perfectly and bypasses the patch.",
        "label_hint": "pro",
    },
    {
        "id": "forum_02",
        "target_claim": "This exploit works reliably on the latest version.",
        "text": "Looks fake. Multiple people report it crashes and does nothing.",
        "label_hint": "anti",
    },
    {
        "id": "forum_03",
        "target_claim": "This exploit works reliably on the latest version.",
        "text": "Not sure. Need more PoC details and verification.",
        "label_hint": "neutral",
    },
]

SOC_TICKET_THREAD = [
    {"role": "user", "text": "We got an alert about possible phishing emails sent to finance staff."},
    {"role": "analyst", "text": "Any IOCs or suspicious links/domains in the email body?"},
    {"role": "user", "text": "Yes, it asks them to verify account urgently and includes a link to secure-login.example.com."},
    {"role": "analyst", "text": "Got it. Please isolate impacted machines and reset credentials."},
    {"role": "user", "text": "Still not fixed. It keeps happening today."},  # needs context from prior turns
]


# -----------------------------
# 1) Pragmatic cues: urgency/pressure/social-engineering signals
# -----------------------------
URGENCY_PATTERNS = [
    r"\burgent\b", r"\bimmediately\b", r"\baction required\b", r"\bnow\b",
    r"\bwithin\b\s+\d+\s*(minutes?|hours?|days?)\b",
    r"\baccount will be (locked|suspended|disabled)\b",
    r"\blast warning\b", r"\bfinal notice\b", r"\bverify\b", r"\bconfirm\b",
    r"\breset (your )?password\b",
]
THREATENING_PATTERNS = [
    r"\bwill be locked\b", r"\bwill be suspended\b", r"\bwill be terminated\b",
    r"\bloss\b", r"\bpenalty\b", r"\blegal action\b",
]

URGENCY_RE = re.compile("|".join(f"({p})" for p in URGENCY_PATTERNS), re.IGNORECASE)
THREAT_RE = re.compile("|".join(f"({p})" for p in THREATENING_PATTERNS), re.IGNORECASE)

def pragmatic_cue_scores(text: str) -> Dict[str, Any]:
    """
    Returns simple pragmatic scores often useful in phishing/social-engineering detection.
    """
    hits_urgency = URGENCY_RE.findall(text)
    hits_threat = THREAT_RE.findall(text)
    exclam = text.count("!")
    caps_words = len(re.findall(r"\b[A-Z]{3,}\b", text))  # "URGENT", etc.

    return {
        "urgency_hits": len(hits_urgency),
        "threat_hits": len(hits_threat),
        "exclamation_count": exclam,
        "caps_words_count": caps_words,
        "has_url": bool(re.search(r"https?://", text, flags=re.IGNORECASE)),
    }


# -----------------------------
# 2) Transformers helpers
# -----------------------------
def load_pipelines() -> Dict[str, Any]:
    from transformers import pipeline

    # Sentiment: multilingual option (better for Thai/EN mix)
    sentiment = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        truncation=True,
    )

    # Zero-shot for stance/intent (works cross-domain reasonably)
    # If your environment is slow, switch to a smaller zero-shot model.
    zeroshot = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        truncation=True,
    )
    return {"sentiment": sentiment, "zeroshot": zeroshot}


# -----------------------------
# 3) Sentiment + urgency demo (phishing-like emails)
# -----------------------------
def analyze_phishing_style_text(pipes: Dict[str, Any], items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for it in items:
        text = it["text"]
        cues = pragmatic_cue_scores(text)
        sent = pipes["sentiment"](text)[0]  # label + score

        # A simple "social-engineering pressure" score (toy)
        pressure_score = (
            cues["urgency_hits"] * 1.0
            + cues["threat_hits"] * 2.0
            + cues["exclamation_count"] * 0.5
            + cues["caps_words_count"] * 0.5
            + (1.0 if cues["has_url"] else 0.0)
        )

        out.append({
            "id": it["id"],
            "label_hint": it.get("label_hint"),
            "sentiment_label": sent["label"],
            "sentiment_score": float(sent["score"]),
            "pressure_score": float(pressure_score),
            **cues,
            "text": text,
        })
    return out


# -----------------------------
# 4) Stance detection demo (toward a claim)
# -----------------------------
def stance_detection(pipes: Dict[str, Any], posts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    labels = ["pro", "anti", "neutral"]
    out = []
    for p in posts:
        claim = p["target_claim"]
        text = p["text"]

        # We frame stance as: does the text support/oppose the claim?
        premise = f"CLAIM: {claim}\nPOST: {text}"

        res = pipes["zeroshot"](premise, candidate_labels=labels, multi_label=False)
        out.append({
            "id": p["id"],
            "label_hint": p.get("label_hint"),
            "pred_label": res["labels"][0],
            "pred_score": float(res["scores"][0]),
            "scores": dict(zip(res["labels"], map(float, res["scores"]))),
            "claim": claim,
            "text": text,
        })
    return out


# -----------------------------
# 5) Intent classification for SOC tickets (triage/routing)
# -----------------------------
SOC_INTENT_LABELS = [
    "phishing investigation",
    "malware investigation",
    "account compromise",
    "network intrusion",
    "vulnerability management",
    "false positive / benign",
    "request for guidance / next steps",
    "status update / follow-up",
]

def classify_soc_intent(pipes: Dict[str, Any], text: str) -> Dict[str, Any]:
    res = pipes["zeroshot"](text, candidate_labels=SOC_INTENT_LABELS, multi_label=True)
    # Sort labels by score (already sorted in pipeline output)
    top = list(zip(res["labels"][:4], [float(s) for s in res["scores"][:4]]))
    return {
        "top_intents": top,
        "all_scores": dict(zip(res["labels"], map(float, res["scores"]))),
    }


# -----------------------------
# 6) Conversation context resolution (pragmatics in SOC threads)
# -----------------------------
FOLLOWUP_PAT = re.compile(r"(?i)\b(still|again|not fixed|keeps happening|same issue|any update|today)\b")

def resolve_followup_with_context(thread: List[Dict[str, str]], window: int = 3) -> List[Dict[str, Any]]:
    """
    If a message looks like a follow-up, we attach a short context summary made from previous turns.
    This is a practical pragmatic step in SOC ticket routing.
    """
    resolved = []
    for i, msg in enumerate(thread):
        text = msg["text"]
        is_followup = bool(FOLLOWUP_PAT.search(text)) and i > 0

        ctx_turns = thread[max(0, i-window):i] if is_followup else []
        ctx_text = " | ".join([f"{t['role']}: {t['text']}" for t in ctx_turns])

        resolved.append({
            "role": msg["role"],
            "text": text,
            "is_followup": is_followup,
            "context_used": ctx_text if is_followup else None,
            "resolved_text_for_model": (ctx_text + "\nFOLLOW-UP: " + text) if is_followup else text,
        })
    return resolved


# -----------------------------
# 7) Run demo
# -----------------------------
def main():
    pipes = load_pipelines()

    print("=" * 90)
    print("A) Pragmatic Processing for phishing/social engineering: Sentiment + Urgency/Pressure cues")
    print("=" * 90)
    phish_out = analyze_phishing_style_text(pipes, PHISHING_EMAILS)
    for r in phish_out:
        print(f"\n[{r['id']}] hint={r['label_hint']}")
        print("  sentiment:", r["sentiment_label"], f"{r['sentiment_score']:.3f}")
        print("  pressure_score:", f"{r['pressure_score']:.2f}",
              "| urgency_hits:", r["urgency_hits"],
              "| threat_hits:", r["threat_hits"],
              "| caps_words:", r["caps_words_count"],
              "| exclam:", r["exclamation_count"],
              "| has_url:", r["has_url"])
        print("  text:", r["text"])

    print("\n" + "=" * 90)
    print("B) Stance Detection (Threat forum / credibility signals)")
    print("=" * 90)
    stance_out = stance_detection(pipes, THREAT_FORUM_POSTS)
    for r in stance_out:
        print(f"\n[{r['id']}] hint={r['label_hint']}")
        print("  pred:", r["pred_label"], f"{r['pred_score']:.3f}")
        print("  scores:", r["scores"])
        print("  claim:", r["claim"])
        print("  post:", r["text"])

    print("\n" + "=" * 90)
    print("C) SOC Ticket Pragmatics: follow-up resolution + intent classification (routing)")
    print("=" * 90)
    resolved = resolve_followup_with_context(SOC_TICKET_THREAD, window=3)

    for m in resolved:
        print(f"\nROLE={m['role']} followup={m['is_followup']}")
        if m["context_used"]:
            print("  context_used:", m["context_used"])
        print("  text:", m["text"])

        # classify intent on resolved text (context-aware for follow-ups)
        intent = classify_soc_intent(pipes, m["resolved_text_for_model"])
        print("  top_intents:", intent["top_intents"])

    print("\nDone.")


if __name__ == "__main__":
    main()
