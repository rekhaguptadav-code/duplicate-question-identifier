import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from datasets import load_dataset


class QuoraDuplicateDetector:
    def __init__(self, threshold=0.75):
        print("Loading embedding model (CPU)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold
        self.index = None
        self.questions = []

    # --------------------------------------------------
    # LOAD QUORA DATASET (YOUR SCHEMA: anchor/positive)
    # --------------------------------------------------
    def load_quora_dataset(self, limit=None):
        print("Loading Quora dataset...")

        dataset = load_dataset(
            "sentence-transformers/quora-duplicates",
            "pair"
        )["train"]

        print("Dataset columns:", dataset.column_names)

        questions = []

        for item in dataset:
            q1 = item["anchor"]
            q2 = item["positive"]

            questions.append(q1)
            questions.append(q2)

            if limit and len(questions) >= limit:
                break

        questions = list(set(questions))

        print(f"Total unique questions loaded: {len(questions)}")
        return questions

    # --------------------------------------------------
    # BUILD FAISS IVF INDEX
    # --------------------------------------------------
    def build_index(self, questions, batch_size=512):
        self.questions = questions

        print("Generating embeddings in batches...")
        embeddings = []

        for i in tqdm(range(0, len(questions), batch_size)):
            batch = questions[i:i + batch_size]
            emb = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(emb)

        embeddings = np.vstack(embeddings)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        print("Building FAISS IVF index...")

        nlist = 100
        quantizer = faiss.IndexFlatIP(dimension)

        self.index = faiss.IndexIVFFlat(
            quantizer,
            dimension,
            nlist,
            faiss.METRIC_INNER_PRODUCT
        )

        self.index.train(embeddings)
        self.index.add(embeddings)

        print("Index built successfully.")
        print("Total vectors in index:", self.index.ntotal)

    # --------------------------------------------------
    # SEARCH DUPLICATE
    # --------------------------------------------------
    def search(self, query):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)

        self.index.nprobe = 10

        similarity, index = self.index.search(query_emb, k=1)

        score = similarity[0][0]
        matched_question = self.questions[index[0][0]]

        if score >= self.threshold:
            return True, matched_question, float(score)
        else:
            return False, None, float(score)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    detector = QuoraDuplicateDetector(threshold=0.75)

    # Start small for testing
    questions = detector.load_quora_dataset(limit=10000)

    detector.build_index(questions)

    print("\nTesting duplicate detection:\n")

    test_query1 = "How can I increase my IQ?"
    result1 = detector.search(test_query1)

    print("Query:", test_query1)
    print("Result:", result1)

    print("\n---------------------------------\n")

    test_query2 = "How do I become more intelligent?"
    result2 = detector.search(test_query2)

    print("Query:", test_query2)
    print("Result:", result2)