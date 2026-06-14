
# Information Retrieval Assignment Report

## 1. Objective
Design and implement an end-to-end IR system using Streamlit where users can:
- Upload/view documents
- Run preprocessing and indexing
- Execute normal and phrase queries
- Compare dictionary structures (BST vs B-Tree)
- Test tolerant retrieval methods

## 2. Dataset Used
Bundled dataset in `data/docs` with 8 text documents focused on IR concepts.

## 3. Implementation in Virtual Lab
Implemented a full Streamlit application (`app.py`) with all tasks accessible from the front end.

### A) Streamlit Workflow
- Dataset source selection: bundled or upload `.txt`
- Document visualization table
- Query and phrase query inputs
- Retrieval and comparison outputs shown in UI

### B) Text Preprocessing
Implemented and displayed:
- Tokenization
- Inverted index creation
- Lowercasing
- Stop-word removal
- Hyphen handling
- Stemming / Lemmatization

#### Stemming vs Lemmatization Evaluation
Used retrieval quality measure: **Average Precision@3** over labeled sample queries (`data/relevance_judgments.csv`).

Observed result (sample run):
- Lemmatization generally performed equal or slightly better than stemming on this dataset, because terms remain more interpretable and less over-truncated.

### C) Phrase Query Processing
Implemented both:
- Biword index
- Positional index

Displayed in UI:
- Biword representation (via pair-based matching)
- Positional representation (term positions)
- Query results from each
- Candidate false positives from biword (`biword_results - positional_results`)

Inference:
- Positional index gave more accurate phrase matches because it enforces exact adjacent positions.
- Biword can produce false positives for longer phrases when all pair components exist but not in one continuous span.

### D) Dictionary Search using BST and B-Tree
Implemented custom:
- Binary Search Tree
- B-Tree

Benchmark:
- Multiple queries
- Repeated lookups
- Average query search time (microseconds)
- Output shown in table in Streamlit

Inference:
- B-Tree often wins on larger dictionaries due to lower tree height and better branching.
- On very small vocabularies, differences may be small.

### E) Tolerant Retrieval
Implemented and demonstrated:
- Wildcard queries (`*`)
- Spelling correction (`difflib` close matches)
- Edit-distance correction (Levenshtein)
- K-gram index candidates
- Phonetic correction (Soundex)

Inference:
- Combining multiple tolerant methods improves robustness to user typos and uncertain spellings.

## 4. Experimental Results
Experimental outputs are generated live in Streamlit and shown as tables in:
- Preprocessing comparison section (Precision@3)
- BST vs B-Tree benchmark section
- Tolerant retrieval candidate lists

> Insert screenshots from your run in the virtual lab below.

## 5. Inference and Discussion (Compulsory)
1. **Which preprocessing technique improved retrieval quality?**  
   Lowercasing + stop-word removal + hyphen normalization improved consistency and reduced vocabulary noise.

2. **Was stemming or lemmatization better?**  
   On the sample dataset, lemmatization was slightly better or tied by Precision@3.

3. **Which phrase index was more accurate?**  
   Positional index.

4. **Which tree structure was faster?**  
   Typically B-Tree for larger dictionaries; verify with benchmark table in your run.

5. **How tolerant was the retrieval model?**  
   Good tolerance for typos/wildcards through edit distance, k-gram, and phonetic candidate generation.

6. **What are the limitations?**  
   Small dataset, lexical ranking only, no neural reranker, no persistent scalable index.

7. **How can the system be improved?**  
   Add BM25, embedding-based retrieval, feedback loops, larger benchmark corpus, and offline index persistence.

## 6. Submission Outputs Included
- Streamlit app code: `app.py`
- Supporting files: dataset + relevance judgments
- Report: this file
- Demo evidence template: `Demo_Evidence.md`
- README with install/run steps
