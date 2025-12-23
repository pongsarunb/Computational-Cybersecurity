"""
Text Processing for Cybersecurity (Example)
- Tokenization (EN/TH)
- POS Tagging + Lemmatization (EN via spaCy)
- Cyber-NER (rule/regex-based): IP, Domain, URL, CVE, Hash (MD5/SHA1/SHA256), Email, File path
- Simple feature extraction for phishing/log analysis

Install:
  pip install -U spacy pythainlp
  python -m spacy download en_core_web_sm
Run:
  python text_processing_cyber.py
"""

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import spacy
from pythainlp import word_tokenize


# -----------------------------
# 1) Cyber Entity Patterns (Regex NER)
# -----------------------------
CYBER_PATTERNS: Dict[str, re.Pattern] = {
    "CVE": re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE),
    "IPV4": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
    ),
    "DOMAIN": re.compile(
        r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:[a-z]{2,63})\b",
        re.IGNORECASE,
    ),
    "URL": re.compile(r"\bhttps?://[^\s)]+", re.IGNORECASE),
    "EMAIL": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63}\b", re.IGNORECASE),
    "MD5": re.compile(r"\b[a-f0-9]{32}\b", re.IGNORECASE),
    "SHA1": re.compile(r"\b[a-f0-9]{40}\b", re.IGNORECASE),
    "SHA256": re.compile(r"\b[a-f0-9]{64}\b", re.IGNORECASE),
    "WIN_PATH": re.compile(r"\b[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*\b"),
    "NIX_PATH": re.compile(r"\b/(?:[^/\s]+/)*[^/\s]*\b"),
    "SQLI_HINT": re.compile(r"(?i)\b(or|and)\b\s+\d+\s*=\s*\d+|(?i)union\s+select|(?i)select\s+\*|--|/\*|\*/"),
    "XSS_HINT": re.compile(r"(?i)<\s*script|onerror\s*=|onload\s*="),
}


@dataclass
class CyberEntity:
    label: str
    text: str
    start: int
    end: int


def extract_cyber_entities(text: str) -> List[CyberEntity]:
    """Regex-based Cyber NER."""
    entities: List[CyberEntity] = []
    for label, pat in CYBER_PATTERNS.items():
        for m in pat.finditer(text):
            entities.append(CyberEntity(label=label, text=m.group(0), start=m.start(), end=m.end()))
    # De-duplicate overlapping identical spans (simple)
    uniq = {}
    for e in entities:
        key = (e.label, e.start, e.end, e.text)
        uniq[key] = e
    return sorted(uniq.values(), key=lambda x: (x.start, x.end, x.label))


# -----------------------------
# 2) spaCy NLP (EN)
# -----------------------------
def load_spacy_en() -> Any:
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model not found. Run:\n  python -m spacy download en_core_web_sm"
        )


def spacy_text_processing(nlp, text: str) -> Dict[str, Any]:
    """Tokenization + POS + Lemma + spaCy NER (general)."""
    doc = nlp(text)
    tokens = [t.text for t in doc]
    pos = [(t.text, t.pos_) for t in doc]
    lemmas = [(t.text, t.lemma_) for t in doc]
    ner = [(ent.text, ent.label_) for ent in doc.ents]
    return {"tokens": tokens, "pos": pos, "lemmas": lemmas, "spacy_ner": ner}


# -----------------------------
# 3) Thai Tokenization (TH)
# -----------------------------
def thai_tokenization(text: str) -> List[str]:
    return word_tokenize(text, engine="newmm")


# -----------------------------
# 4) Simple feature extraction for security text
# -----------------------------
PHISHING_CUES = [
    r"verify your account", r"urgent", r"immediately", r"password", r"reset",
    r"click", r"login", r"confirm", r"payment", r"invoice", r"bank", r"security alert",
]
PHISHING_PAT = re.compile("|".join(f"({c})" for c in PHISHING_CUES), re.IGNORECASE)

def extract_security_features(text: str) -> Dict[str, Any]:
    ents = extract_cyber_entities(text)
    return {
        "len_chars": len(text),
        "len_words_rough": len(re.findall(r"\w+", text)),
        "has_url": any(e.label == "URL" for e in ents),
        "has_domain": any(e.label == "DOMAIN" for e in ents),
        "has_ip": any(e.label == "IPV4" for e in ents),
        "has_cve": any(e.label == "CVE" for e in ents),
        "has_hash": any(e.label in {"MD5", "SHA1", "SHA256"} for e in ents),
        "has_sqli_hint": any(e.label == "SQLI_HINT" for e in ents),
        "has_xss_hint": any(e.label == "XSS_HINT" for e in ents),
        "phishing_cue_hits": len(PHISHING_PAT.findall(text)),
        "entities": [asdict(e) for e in ents],
    }


# -----------------------------
# 5) Demo data (logs / phishing email / threat report)
# -----------------------------
SAMPLES = {
    "web_log_sqli": r'192.168.1.10 - - [19/Dec/2025:10:15:03 +0700] "GET /login.php?user=admin&id=1 OR 1=1 -- HTTP/1.1" 200 532 "-" "Mozilla/5.0"',
    "threat_report": "Emotet malware exploited CVE-2023-1234 and contacted hxxp://malicious-example[.]com. "
                     "Indicator: 8.8.8.8, sha256=3b7f0f2a6f0a7f9b2d3c4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a.",
    "phishing_email": "Subject: URGENT - Verify your account immediately\n"
                      "Dear user, click https://secure-login.example.com/reset to confirm your password.\n"
                      "If you do not verify, your account will be locked.\n"
                      "Contact: support@secure-login.example.com",
    "thai_ticket": "แจ้งเหตุ: ผู้ใช้ได้รับอีเมลให้คลิกลิงก์เพื่อยืนยันบัญชีและรีเซ็ตรหัสผ่าน "
                   "โดเมนที่พบ secure-login.example.com และมี IP 10.0.0.5 ใน log",
}


# -----------------------------
# 6) Main
# -----------------------------
def main():
    nlp = load_spacy_en()

    print("=" * 80)
    print("A) Text Processing (EN) + Cyber NER + Features")
    print("=" * 80)
    for name, text in SAMPLES.items():
        print(f"\n--- SAMPLE: {name} ---")
        print(text)

        # EN processing with spaCy (works for English-like text)
        en_out = spacy_text_processing(nlp, text)
        print("\n[TOKENS] (first 25):", en_out["tokens"][:25])
        print("[POS] (first 15):", en_out["pos"][:15])
        print("[LEMMAS] (first 15):", en_out["lemmas"][:15])
        print("[spaCy NER]:", en_out["spacy_ner"])

        # Cyber-specific NER (regex)
        feats = extract_security_features(text)
        print("\n[CYBER ENTITIES]:")
        for e in feats["entities"]:
            print(" ", e)
        print("[FEATURES]:",
              {k: v for k, v in feats.items() if k != "entities"})

    print("\n" + "=" * 80)
    print("B) Thai Tokenization (TH) + Cyber Entities")
    print("=" * 80)
    th = SAMPLES["thai_ticket"]
    print("\n--- SAMPLE: thai_ticket ---")
    print(th)
    th_tokens = thai_tokenization(th)
    print("[TH TOKENS]:", th_tokens)

    th_feats = extract_security_features(th)
    print("[CYBER ENTITIES]:")
    for e in th_feats["entities"]:
        print(" ", e)


if __name__ == "__main__":
    main()
