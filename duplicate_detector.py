import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ============================================================
# 1. DOMAIN + INTENT DETECTION
# ============================================================

def detect_domain(question):
    q = question.lower()
    if "d/dx" in q or "derivative" in q:
        return "calculus"
    if "speed" in q or "distance" in q or "time" in q or "m/s" in q:
        return "physics"
    return "general"

def extract_intent(question):
    q = question.lower()
    if "distance" in q:
        return "distance"
    if "time" in q:
        return "time"
    if "speed" in q:
        return "speed"
    if "d/dx" in q or "derivative" in q:
        return "derivative"
    return "unknown"

# ============================================================
# 2. CANONICALIZATION
# ============================================================

def normalize_numbers(text):
    return re.sub(r"\d+(\.\d+)?", "NUM", text)

def canonicalize(question):
    domain = detect_domain(question)
    intent = extract_intent(question)
    q = question.lower()

    # -------- CALCULUS HANDLING --------
    if domain == "calculus":
        # Normalize exponent
        structure = re.sub(r"\^\d+", "^N", q)

    # -------- PHYSICS HANDLING --------
    elif domain == "physics":

        has_speed = "m/s" in q or "speed" in q
        has_time = "minute" in q or "time" in q
        has_distance = "distance" in q

        if has_speed and has_time and has_distance:
            structure = "object moves with speed S for time T. find distance."
        elif has_speed and has_distance and has_time:
            structure = "object moves with speed S for time T. find time."
        elif has_speed and has_distance:
            structure = "object moves with speed S for distance D."
        else:
            structure = normalize_numbers(q)

    # -------- GENERAL --------
    else:
        structure = normalize_numbers(q)

    canonical_form = f"domain:{domain} | intent:{intent} | structure:{structure}"

    return {
        "canonical": canonical_form,
        "domain": domain,
        "intent": intent,
        "raw": question
    }

# ============================================================
# 3. EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    return model.encode(text, normalize_embeddings=True)

# ============================================================
# 4. VECTOR DATABASE (FAISS)
# ============================================================

dimension = 384
index = faiss.IndexFlatIP(dimension)
metadata_store = []

def add_question(question):
    data = canonicalize(question)
    vector = embed(data["canonical"])
    index.add(np.array([vector]))
    metadata_store.append(data)

# ============================================================
# 5. DUPLICATE DETECTOR
# ============================================================

def find_duplicate(question, threshold=0.80, strict=False):
    data = canonicalize(question)
    vector = embed(data["canonical"])

    if index.ntotal == 0:
        return False, None

    D, I = index.search(np.array([vector]), k=5)

    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue

        existing = metadata_store[idx]

        if score > threshold:

            # Intent must match
            if existing["intent"] != data["intent"]:
                continue

            # Strict mode
            if strict:
                if existing["raw"] == data["raw"]:
                    return True, existing
            else:
                return True, existing

    return False, None

# ============================================================
# 6. TESTING (STEP 6)
# ============================================================

if __name__ == "__main__":

    print("\nAdding base questions...")
    add_question("Tom travels with speed 5 m/s for 5 minutes. Find distance.")
    add_question("d/dx(x^3)")

    print("\nTest 1 (Physics template duplicate):")
    print(find_duplicate("Tobey travels at 10 m/s for 10 minutes. Find distance."))

    print("\nTest 2 (Different intent - should be False):")
    print(find_duplicate("Tobey travels at 10 m/s for 10 minutes. Find time."))

    print("\nTest 3 (Calculus template duplicate):")
    print(find_duplicate("d/dx(x^4)"))

    print("\nTest 4 (Strict mode - exact match):")
    print(find_duplicate("d/dx(x^3)", strict=True))

    print("\nTest 5 (Strict mode - different exponent):")
    print(find_duplicate("d/dx(x^4)", strict=True))