import io
import os
import re
import time
import math
import difflib
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set

import pandas as pd
import streamlit as st

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer


# ---------------------------
# NLTK bootstrap
# ---------------------------
NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}

for pkg, resource_path in NLTK_RESOURCES.items():
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(pkg, quiet=True)


# ---------------------------
# Data structures
# ---------------------------
@dataclass
class Document:
    doc_id: str
    text: str


class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None

    def insert(self, key: str):
        if self.root is None:
            self.root = BSTNode(key)
            return
        cur = self.root
        while True:
            if key < cur.key:
                if cur.left is None:
                    cur.left = BSTNode(key)
                    return
                cur = cur.left
            elif key > cur.key:
                if cur.right is None:
                    cur.right = BSTNode(key)
                    return
                cur = cur.right
            else:
                return

    def search(self, key: str) -> bool:
        cur = self.root
        while cur is not None:
            if key == cur.key:
                return True
            cur = cur.left if key < cur.key else cur.right
        return False


class BTreeNode:
    def __init__(self, t: int, leaf: bool):
        self.t = t
        self.leaf = leaf
        self.keys = []
        self.children = []


class BTree:
    def __init__(self, t: int = 3):
        self.t = t
        self.root = BTreeNode(t, True)

    def search(self, key: str, node: BTreeNode = None) -> bool:
        if node is None:
            node = self.root
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and node.keys[i] == key:
            return True
        if node.leaf:
            return False
        return self.search(key, node.children[i])

    def split_child(self, parent: BTreeNode, i: int):
        t = self.t
        y = parent.children[i]
        z = BTreeNode(t, y.leaf)

        median = y.keys[t - 1]
        z.keys = y.keys[t:]
        y.keys = y.keys[: t - 1]

        if not y.leaf:
            z.children = y.children[t:]
            y.children = y.children[:t]

        parent.children.insert(i + 1, z)
        parent.keys.insert(i, median)

    def insert_non_full(self, node: BTreeNode, key: str):
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append("")
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            if i >= 0 and node.keys[i] == key:
                node.keys.pop()
                return
            node.keys[i + 1] = key
            return

        while i >= 0 and key < node.keys[i]:
            i -= 1
        i += 1
        if i < len(node.keys) and node.keys[i] == key:
            return
        if len(node.children[i].keys) == 2 * self.t - 1:
            self.split_child(node, i)
            if key > node.keys[i]:
                i += 1
        self.insert_non_full(node.children[i], key)

    def insert(self, key: str):
        root = self.root
        if len(root.keys) == 2 * self.t - 1:
            new_root = BTreeNode(self.t, False)
            new_root.children.append(root)
            self.split_child(new_root, 0)
            self.root = new_root
            self.insert_non_full(new_root, key)
        else:
            self.insert_non_full(root, key)


# ---------------------------
# IR utility functions
# ---------------------------
def load_default_docs() -> List[Document]:
    docs_dir = Path("data/docs")
    docs = []
    if docs_dir.exists():
        for p in sorted(docs_dir.glob("*.txt")):
            docs.append(Document(doc_id=p.stem, text=p.read_text(encoding="utf-8")))
    return docs


def load_uploaded_docs(uploaded_files) -> List[Document]:
    docs = []
    for f in uploaded_files:
        text = f.read().decode("utf-8", errors="ignore")
        doc_id = Path(f.name).stem
        docs.append(Document(doc_id=doc_id, text=text))
    return docs


def normalize_hyphen(text: str) -> str:
    # state-of-the-art -> state of the art
    return re.sub(r"[-‐‑‒–—]+", " ", text)


def preprocess_text(
    text: str,
    use_lower: bool = True,
    remove_stop: bool = True,
    hyphen: bool = True,
    method: str = "none",
) -> List[str]:
    if hyphen:
        text = normalize_hyphen(text)
    if use_lower:
        text = text.lower()

    tokens = re.findall(r"[A-Za-z0-9]+", text)

    if remove_stop:
        sw = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in sw]

    if method == "stemming":
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    elif method == "lemmatization":
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def build_inverted_index(tokenized_docs: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    idx = defaultdict(dict)
    for d, toks in tokenized_docs.items():
        freq = Counter(toks)
        for term, c in freq.items():
            idx[term][d] = c
    return dict(idx)


def build_positional_index(tokenized_docs: Dict[str, List[str]]) -> Dict[str, Dict[str, List[int]]]:
    idx = defaultdict(lambda: defaultdict(list))
    for d, toks in tokenized_docs.items():
        for pos, term in enumerate(toks):
            idx[term][d].append(pos)
    return {k: dict(v) for k, v in idx.items()}


def build_biword_index(tokenized_docs: Dict[str, List[str]]) -> Dict[Tuple[str, str], Set[str]]:
    idx = defaultdict(set)
    for d, toks in tokenized_docs.items():
        for i in range(len(toks) - 1):
            idx[(toks[i], toks[i + 1])].add(d)
    return dict(idx)


def phrase_query_positional(phrase_tokens: List[str], pos_index) -> Set[str]:
    if not phrase_tokens:
        return set()
    first = phrase_tokens[0]
    if first not in pos_index:
        return set()
    candidate_docs = set(pos_index[first].keys())
    for t in phrase_tokens[1:]:
        if t not in pos_index:
            return set()
        candidate_docs &= set(pos_index[t].keys())

    results = set()
    for d in candidate_docs:
        positions = set(pos_index[first][d])
        for offset, term in enumerate(phrase_tokens[1:], start=1):
            shifted = {p - offset for p in pos_index[term][d]}
            positions &= shifted
        if positions:
            results.add(d)
    return results


def phrase_query_biword(phrase_tokens: List[str], biword_index) -> Set[str]:
    if len(phrase_tokens) < 2:
        return set()
    pairs = [(phrase_tokens[i], phrase_tokens[i + 1]) for i in range(len(phrase_tokens) - 1)]
    doc_sets = []
    for p in pairs:
        if p not in biword_index:
            return set()
        doc_sets.append(biword_index[p])
    return set.intersection(*map(set, doc_sets)) if doc_sets else set()


def tf_idf_rank(query_tokens: List[str], tokenized_docs: Dict[str, List[str]]):
    N = len(tokenized_docs)
    if N == 0:
        return []
    df = Counter()
    tf_docs = {}
    for d, toks in tokenized_docs.items():
        tf = Counter(toks)
        tf_docs[d] = tf
        for term in tf.keys():
            df[term] += 1

    scores = defaultdict(float)
    qtf = Counter(query_tokens)
    for term, qf in qtf.items():
        if df[term] == 0:
            continue
        idf = math.log((N + 1) / (df[term] + 1)) + 1
        for d, tf in tf_docs.items():
            if term in tf:
                scores[d] += (tf[term] * idf) * (qf * idf)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def precision_at_k(ranked_docs: List[str], relevant_set: Set[str], k: int = 5) -> float:
    if k == 0:
        return 0.0
    top = ranked_docs[:k]
    if not top:
        return 0.0
    hit = sum(1 for d in top if d in relevant_set)
    return hit / min(k, len(top))


def evaluate_stem_vs_lemma(docs, use_lower, remove_stop, hyphen):
    qpath = Path("data/relevance_judgments.csv")
    if not qpath.exists():
        return None

    judgements = pd.read_csv(qpath)
    methods = ["stemming", "lemmatization"]
    rows = []
    for method in methods:
        tokenized = {
            d.doc_id: preprocess_text(
                d.text,
                use_lower=use_lower,
                remove_stop=remove_stop,
                hyphen=hyphen,
                method=method,
            )
            for d in docs
        }

        scores = []
        for _, row in judgements.iterrows():
            q = str(row["query"])
            rel = set(str(row["relevant_docs"]).split(";"))
            q_toks = preprocess_text(
                q,
                use_lower=use_lower,
                remove_stop=remove_stop,
                hyphen=hyphen,
                method=method,
            )
            ranked = [d for d, _ in tf_idf_rank(q_toks, tokenized)]
            scores.append(precision_at_k(ranked, rel, k=3))

        rows.append({"Method": method, "Avg Precision@3": round(sum(scores) / len(scores), 4)})

    out = pd.DataFrame(rows).sort_values("Avg Precision@3", ascending=False)
    return out


def wildcard_search(pattern: str, vocabulary: List[str]) -> List[str]:
    regex = "^" + re.escape(pattern).replace("\\*", ".*") + "$"
    r = re.compile(regex)
    return [t for t in vocabulary if r.match(t)]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def soundex(word: str) -> str:
    word = word.upper()
    if not word:
        return ""
    mappings = {
        "B": "1",
        "F": "1",
        "P": "1",
        "V": "1",
        "C": "2",
        "G": "2",
        "J": "2",
        "K": "2",
        "Q": "2",
        "S": "2",
        "X": "2",
        "Z": "2",
        "D": "3",
        "T": "3",
        "L": "4",
        "M": "5",
        "N": "5",
        "R": "6",
    }
    first = word[0]
    digits = [mappings.get(ch, "") for ch in word[1:]]
    filtered = []
    prev = ""
    for d in digits:
        if d != prev:
            filtered.append(d)
        if d != "":
            prev = d
    filtered = [d for d in filtered if d != ""]
    code = first + "".join(filtered)
    return (code + "000")[:4]


def build_kgram_index(vocabulary: List[str], k: int = 3):
    idx = defaultdict(set)
    for term in vocabulary:
        padded = f"${term}$"
        for i in range(len(padded) - k + 1):
            idx[padded[i : i + k]].add(term)
    return dict(idx)


def kgram_candidates(term: str, kgram_idx, k: int = 3) -> List[str]:
    padded = f"${term}$"
    grams = set(padded[i : i + k] for i in range(len(padded) - k + 1))
    cands = Counter()
    for g in grams:
        for t in kgram_idx.get(g, []):
            cands[t] += 1
    if not cands:
        return []
    ranked = sorted(cands.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked[:10]]


def benchmark_trees(vocabulary: List[str], queries: List[str], rounds: int = 200):
    bst = BST()
    bt = BTree(t=4)
    for term in vocabulary:
        bst.insert(term)
        bt.insert(term)

    rows = []
    for q in queries:
        # search time benchmark
        t0 = time.perf_counter()
        for _ in range(rounds):
            bst.search(q)
        bst_time = (time.perf_counter() - t0) / rounds * 1e6

        t0 = time.perf_counter()
        for _ in range(rounds):
            bt.search(q)
        bt_time = (time.perf_counter() - t0) / rounds * 1e6

        faster = "BST" if bst_time < bt_time else "B-Tree"
        rows.append(
            {
                "Query": q,
                "BST avg search (µs)": round(bst_time, 2),
                "B-Tree avg search (µs)": round(bt_time, 2),
                "Faster": faster,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="IR End-to-End System", layout="wide")
st.title("End-to-End Information Retrieval System")

st.sidebar.header("Input Data")
source = st.sidebar.radio("Dataset source", ["Use bundled dataset", "Upload .txt files"])
uploaded = st.sidebar.file_uploader(
    "Upload text files",
    type=["txt"],
    accept_multiple_files=True,
    disabled=(source != "Upload .txt files"),
)

if source == "Upload .txt files" and uploaded:
    documents = load_uploaded_docs(uploaded)
else:
    documents = load_default_docs()

if not documents:
    st.warning("No documents loaded. Upload .txt files or add docs in data/docs.")
    st.stop()

st.sidebar.header("Preprocessing")
use_lower = st.sidebar.checkbox("Lowercasing", value=True)
remove_stop = st.sidebar.checkbox("Stop word removal", value=True)
hyphen = st.sidebar.checkbox("Hyphen handling", value=True)
method = st.sidebar.selectbox("Normalization", ["none", "stemming", "lemmatization"])

# Tokenize documents
with st.spinner("Preprocessing documents..."):
    tokenized_docs = {
        d.doc_id: preprocess_text(
            d.text,
            use_lower=use_lower,
            remove_stop=remove_stop,
            hyphen=hyphen,
            method=method,
        )
        for d in documents
    }

inverted_index = build_inverted_index(tokenized_docs)
positional_index = build_positional_index(tokenized_docs)
biword_index = build_biword_index(tokenized_docs)
vocab = sorted(inverted_index.keys())


# A) Workflow + document view
st.header("A) Streamlit Workflow")
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Uploaded / Available Documents")
    doc_df = pd.DataFrame(
        [{"doc_id": d.doc_id, "chars": len(d.text), "preview": d.text[:120] + "..."} for d in documents]
    )
    st.dataframe(doc_df, use_container_width=True)

with col2:
    st.subheader("Tokenization Output (first 25 tokens)")
    token_preview = {d: toks[:25] for d, toks in tokenized_docs.items()}
    st.json(token_preview)


# B) Preprocessing outputs
st.header("B) Text Preprocessing + Inverted Index")
st.write(f"Vocabulary size: **{len(vocab)}**")

sample_terms = st.multiselect("Select terms to inspect postings", options=vocab, default=vocab[: min(8, len(vocab))])
if sample_terms:
    rows = []
    for t in sample_terms:
        rows.append(
            {
                "term": t,
                "document_frequency": len(inverted_index[t]),
                "postings": dict(inverted_index[t]),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.subheader("Stemming vs Lemmatization Comparison")
cmp_df = evaluate_stem_vs_lemma(documents, use_lower, remove_stop, hyphen)
if cmp_df is not None:
    st.dataframe(cmp_df, use_container_width=True)
    best = cmp_df.iloc[0]["Method"]
    st.success(
        f"For the bundled relevance judgments, **{best}** gives higher Avg Precision@3. "
        "Use this as dataset-specific evidence for your report."
    )
else:
    st.info("Add data/relevance_judgments.csv to compute retrieval-quality comparison automatically.")


# C) Phrase query
st.header("C) Phrase Query: Biword vs Positional Index")
phrase = st.text_input("Enter a phrase query", value="phrase query processing")
if phrase.strip():
    q_tokens = preprocess_text(
        phrase,
        use_lower=use_lower,
        remove_stop=remove_stop,
        hyphen=hyphen,
        method=method,
    )
    if len(q_tokens) < 2:
        st.warning("Enter at least two terms for phrase query.")
    else:
        bi_docs = phrase_query_biword(q_tokens, biword_index)
        pos_docs = phrase_query_positional(q_tokens, positional_index)
        fp = sorted(list(bi_docs - pos_docs))

        c1, c2 = st.columns(2)
        with c1:
            st.write("Biword matches", sorted(list(bi_docs)))
        with c2:
            st.write("Positional matches", sorted(list(pos_docs)))

        st.write("Potential biword false positives", fp if fp else "None in current dataset/query")
        st.caption(
            "Biword index checks pair existence but not full positional continuity for all terms; "
            "positional index verifies exact adjacent positions."
        )


# Search panel
st.header("Query Search (TF-IDF)")
query = st.text_input("Enter a search query", value="retrieval quality preprocessing")
if query.strip():
    q_tokens = preprocess_text(
        query,
        use_lower=use_lower,
        remove_stop=remove_stop,
        hyphen=hyphen,
        method=method,
    )
    ranked = tf_idf_rank(q_tokens, tokenized_docs)
    if not ranked:
        st.info("No results found.")
    else:
        rows = []
        for d, s in ranked[:10]:
            text = next(doc.text for doc in documents if doc.doc_id == d)
            rows.append({"doc_id": d, "score": round(s, 4), "snippet": text[:150] + "..."})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# D) BST vs B-Tree benchmarking
st.header("D) Dictionary Search: BST vs B-Tree")
default_queries = ["retrieval", "index", "query", "lemmatization", "phonetic", "nonexistingterm"]
q_input = st.text_input("Benchmark queries (comma-separated)", value=",".join(default_queries))
queries = [q.strip() for q in q_input.split(",") if q.strip()]
if st.button("Run BST vs B-Tree experiment"):
    with st.spinner("Running benchmark..."):
        bench = benchmark_trees(vocab, queries, rounds=300)
    st.dataframe(bench, use_container_width=True)
    winner = bench["Faster"].value_counts().idxmax()
    st.success(f"Overall winner across selected queries: **{winner}**")


# E) Tolerant retrieval
st.header("E) Tolerant Retrieval")
col_w, col_s, col_p = st.columns(3)

with col_w:
    st.subheader("Wildcard")
    pattern = st.text_input("Wildcard pattern (* supported)", value="retriev*")
    if pattern:
        st.write(wildcard_search(pattern.lower(), vocab)[:30])

with col_s:
    st.subheader("Spelling / Edit Distance")
    miss = st.text_input("Possibly misspelled term", value="retrival")
    if miss:
        close = difflib.get_close_matches(miss, vocab, n=5, cutoff=0.6)
        lv = sorted([(t, levenshtein(miss, t)) for t in vocab], key=lambda x: x[1])[:5]
        st.write("Close matches", close)
        st.write("Best edit-distance candidates", lv)

with col_p:
    st.subheader("K-gram + Phonetic")
    term = st.text_input("Term for K-gram / phonetic", value="foneetic")
    if term:
        kidx = build_kgram_index(vocab, k=3)
        kg = kgram_candidates(term.lower(), kidx, k=3)[:8]
        sx = soundex(term)
        phon = [t for t in vocab if soundex(t) == sx][:10]
        st.write("K-gram candidates", kg)
        st.write(f"Soundex({term}) = {sx}")
        st.write("Phonetic candidates", phon)


# G) Inference section
st.header("G) Inference and Discussion")
with st.expander("Auto-generated inferences (edit in your report as needed)", expanded=True):
    st.markdown(
        """
1. **Preprocessing impact**: Lowercasing + stop word removal usually improves precision by reducing noise and vocabulary mismatch.
2. **Stemming vs lemmatization**: Use the table above (`Avg Precision@3`) as experimental evidence for this dataset.
3. **Phrase query accuracy**: Positional index is more accurate because it validates exact adjacency positions.
4. **Tree performance**: Benchmark table shows whether `BST` or `B-Tree` is faster for current vocabulary and query set.
5. **Tolerant behavior**: Wildcard, edit distance, k-gram, and phonetic options recover useful candidates for imperfect queries.
6. **Limitations**: Small dataset, lexical matching only, no learning-to-rank, and no large-scale optimization.
7. **Improvements**: Add BM25, semantic embeddings, relevance feedback, persistent index storage, and larger benchmark datasets.
        """
    )

st.caption("Tip: Capture screenshots from each section for your submission report and demo evidence.")
