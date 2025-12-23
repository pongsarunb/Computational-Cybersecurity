"""
Semantic Processing for Cybersecurity (Example Pipeline)
Covers:
  1) Word Sense Disambiguation (WSD) using Lesk (NLTK) on ambiguous terms (e.g., "worm", "trojan")
  2) Coreference Resolution using fastcoref (who/what "it", "they" refer to)
  3) Semantic Role Labeling (SRL) using AllenNLP (extract event roles: attacker/action/target/vector)
  4) Semantic Similarity / Threat Correlation using sentence embeddings (Transformers)

Install (recommended):
  pip install -U nltk fastcoref allennlp allennlp-models transformers torch
  python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

Notes:
  - SRL model download happens automatically the first time you run it (cached afterward).
  - Coref/SRL are English-focused in this demo. For Thai, usually need fine-tuned models or rules + NER.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple
import numpy as np

# -----------------------------
# 0) Sample cyber texts (Threat Reports / Tickets)
# -----------------------------
REPORTS = [
    "The attacker exploited CVE-2023-1234. It allowed remote code execution on the hospital network. "
    "They delivered ransomware via a phishing email and encrypted critical servers.",
    "A worm spread laterally after exploiting a vulnerability. It infected endpoints in the finance department "
    "through a malicious attachment and then contacted a C2 domain.",
    "A trojan was found on a user's laptop. It stole credentials and exfiltrated data to an external IP address.",
]

AMBIGUOUS_TERMS = ["worm", "trojan", "virus", "bank"]


# -----------------------------
# 1) WSD (Word Sense Disambiguation) - NLTK Lesk
# -----------------------------
def wsd_lesk(sentence: str, target: str) -> Dict[str, Any]:
    """
    Returns best WordNet sense for target in sentence using Lesk algorithm (toy baseline).
    Good for demonstrating ambiguity like worm (animal vs malware) / trojan (people vs malware).
    """
    import nltk
    from nltk.wsd import lesk
    from nltk.corpus import wordnet as wn

    tokens = re.findall(r"[A-Za-z]+", sentence.lower())
    sense = lesk(tokens, target.lower(), pos="n")  # try noun
    if sense is None:
        return {"target": target, "sense": None, "definition": None, "synset": None}

    return {
        "target": target,
        "synset": sense.name(),
        "definition": sense.definition(),
        "lemmas": [l.name() for l in sense.lemmas()],
    }


# -----------------------------
# 2) Coreference Resolution - fastcoref
# -----------------------------
def resolve_coref(text: str) -> List[List[str]]:
    """
    Returns coreference clusters as strings.
    Example: ["the attacker", "they"] ; ["CVE-2023-1234", "it"] (sometimes)
    """
    from fastcoref import FCoref

    coref = FCoref(device="cpu")
    pred = coref.predict(texts=[text])[0]
    return pred.get_clusters(as_strings=True)


# -----------------------------
# 3) Semantic Role Labeling (SRL) - AllenNLP
# -----------------------------
def srl_frames(text: str) -> List[Dict[str, Any]]:
    """
    Returns SRL frames (verbs + arguments) for each sentence.
    We use a BERT-based SRL model from AllenNLP.
    """
    from allennlp.predictors.predictor import Predictor
    import allennlp_models.tagging  # noqa: F401 (registers models)

    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
    )

    # Split sentences crudely for demo
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    outputs = []
    for s in sents:
        if not s:
            continue
        res = predictor.predict(sentence=s)
        outputs.append({"sentence": s, "verbs": res.get("verbs", [])})
    return outputs


# -----------------------------
# 4) Sentence Embeddings for Threat Correlation (Transformers)
# -----------------------------
def mean_pooling(token_embs: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Mean pooling with attention mask.
    token_embs: [1, seq_len, hidden]
    attention_mask: [1, seq_len]
    """
    mask = attention_mask[..., None]  # [1, seq_len, 1]
    summed = (token_embs * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    return summed / counts


def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """
    Produces sentence embeddings for texts using a transformer encoder.
    Uses AutoTokenizer + AutoModel.
    """
    from transformers import AutoTokenizer, AutoModel
    import torch

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()

    embs = []
    with torch.no_grad():
        for t in texts:
            batch = tok(t, return_tensors="pt", truncation=True, max_length=256, padding=True)
            out = mdl(**batch)
            token_embs = out.last_hidden_state.cpu().numpy()
            attn = batch["attention_mask"].cpu().numpy()
            sent_emb = mean_pooling(token_embs, attn)[0]
            # L2 normalize
            sent_emb = sent_emb / (np.linalg.norm(sent_emb) + 1e-12)
            embs.append(sent_emb)
    return np.vstack(embs)


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    return X @ X.T  # since normalized


# -----------------------------
# 5) Cyber Semantic Extraction Helpers (lightweight)
# -----------------------------
CYBER_ROLE_HINTS = {
    "ATTACK_VECTOR": re.compile(r"(?i)\b(phishing email|malicious attachment|exploit|drive-by|spearphish)\b"),
    "IMPACT": re.compile(r"(?i)\b(encrypt(ed|ion)|exfiltrat(ed|ion)|credential theft|rce|remote code execution)\b"),
    "TARGET": re.compile(r"(?i)\b(hospital network|finance department|endpoints|servers|laptop|workstation)\b"),
}

def quick_semantic_hints(text: str) -> Dict[str, List[str]]:
    hints = {}
    for k, pat in CYBER_ROLE_HINTS.items():
        hints[k] = list({m.group(0) for m in pat.finditer(text)})
    return hints


# -----------------------------
# Main demo
# -----------------------------
def main():
    print("=" * 90)
    print("A) WSD (Lesk) on ambiguous terms inside cyber reports")
    print("=" * 90)
    for i, rep in enumerate(REPORTS, 1):
        print(f"\n[Report {i}] {rep}")
        for term in AMBIGUOUS_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", rep, flags=re.IGNORECASE):
                wsd = wsd_lesk(rep, term)
                print("  WSD:", wsd)

    print("\n" + "=" * 90)
    print("B) Coreference Resolution (fastcoref)")
    print("=" * 90)
    for i, rep in enumerate(REPORTS, 1):
        try:
            clusters = resolve_coref(rep)
            print(f"\n[Report {i}] Coref clusters:")
            for c in clusters:
                print(" ", c)
        except Exception as e:
            print(f"\n[Report {i}] Coref failed:", repr(e))

    print("\n" + "=" * 90)
    print("C) Semantic Role Labeling (AllenNLP SRL)")
    print("=" * 90)
    for i, rep in enumerate(REPORTS, 1):
        try:
            frames = srl_frames(rep)
            print(f"\n[Report {i}]")
            for item in frames:
                print(" Sentence:", item["sentence"])
                # Print compact SRL descriptions
                for v in item["verbs"]:
                    # v['description'] example: [ARG0: The attacker] [V: exploited] [ARG1: CVE-...]
                    print("  SRL:", v.get("description"))
        except Exception as e:
            print(f"\n[Report {i}] SRL failed:", repr(e))

    print("\n" + "=" * 90)
    print("D) Semantic Similarity / Threat Correlation (Sentence Embeddings)")
    print("=" * 90)
    try:
        embs = embed_texts(REPORTS)
        sim = cosine_sim_matrix(embs)
        print("Cosine similarity matrix (higher = more semantically similar):\n", np.round(sim, 3))

        # show top related pairs (excluding diagonal)
        pairs: List[Tuple[float, int, int]] = []
        n = sim.shape[0]
        for a in range(n):
            for b in range(a + 1, n):
                pairs.append((float(sim[a, b]), a, b))
        pairs.sort(reverse=True)

        print("\nTop similar report pairs:")
        for score, a, b in pairs[:3]:
            print(f"  Report {a+1} ↔ Report {b+1}: score={score:.3f}")
            print("   -", REPORTS[a])
            print("   -", REPORTS[b])
    except Exception as e:
        print("Embedding similarity failed:", repr(e))

    print("\n" + "=" * 90)
    print("E) Quick semantic hints (lightweight cyber role cues)")
    print("=" * 90)
    for i, rep in enumerate(REPORTS, 1):
        print(f"\n[Report {i}] Hints:", quick_semantic_hints(rep))


if __name__ == "__main__":
    main()
