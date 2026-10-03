# Tips Hindawi University — RAG Lab

## Overview

This repository contains the RAG implementations developed for the **Tips Hindawi University information** lab.

The project starts from the lecture's simple RAG architecture and then develops more controlled English and Arabic implementations using modern Hugging Face models and multilingual embeddings.

The core idea is:

```text
University PDF
      ↓
Text Extraction
      ↓
Chunking
      ↓
Embeddings
      ↓
FAISS Vector Search
      ↓
Question
      ↓
Relevant Context
      ↓
LLM Prompt
      ↓
Generated Answer
```

The project demonstrates an important RAG concept:

> Retrieval can provide relevant evidence, but retrieval alone does not guarantee that the final LLM answer is fully grounded in that evidence.

---

# 1. Repository Goals

The project has several goals:

1. Reproduce the basic RAG architecture from the lecture.
2. Build an English RAG baseline.
3. Improve the English RAG implementation with stronger retrieval and grounding controls.
4. Build an Arabic RAG implementation using the same multilingual architecture.
5. Compare model choices, embeddings, chunking, retrieval, prompting, and generation behavior.
6. Inspect the retrieved evidence instead of evaluating only the final answer.
7. Document both successful behavior and failure cases.

---

# 2. Available Notebooks

The current project contains **three RAG notebooks** in the Library:

| Notebook | Role |
|---|---|
| `Tips-Hindawi-English-Rag-Task_.ipynb` | English baseline |
| `Tips-Hindawi-English-Rag-Task-1_.ipynb` | Improved English RAG |
| `Tips-Hindawi-Arabic-Rag-Task_.ipynb` | Arabic RAG |

---

# 3. Project Structure

The intended GitHub organization is:

```text
Tips-Hindawi-RAG/
│
├── README.md
│
├── Tips-Hindawi-English-Rag-Task_.ipynb
├── README_Tips-Hindawi-English-Rag-Task.md
│
├── Tips-Hindawi-English-Rag-Task-1_.ipynb
├── README_Tips-Hindawi-English-Rag-Task-1.md
│
├── Tips-Hindawi-Arabic-Rag-Task_.ipynb
└──  README_Tips-Hindawi-Arabic-Rag-Task.md
```
---

# 4. Original Lecture Pipeline

The lecture notebook introduces the basic pipeline:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS
 ↓
Retrieval
 ↓
Prompt
 ↓
Language Model
 ↓
Answer
```

The lecture implementation uses:

- `Mistral-Nemo-Instruct-2407`
- `all-MiniLM-L6-v2`
- FAISS `IndexFlatL2`
- PyPDF2
- word-based chunking

This lecture structure is the conceptual foundation for the project.

---

# 5. Notebook 1 — English RAG Baseline

File:

```text
Tips-Hindawi-English-Rag-Task_.ipynb
```

README:

```text
README_Tips-Hindawi-English-Rag-Task.md
```

Branch:

```text
english-baseline
```

## Main configuration

| Component | Baseline |
|---|---|
| Generator | `mistralai/Mistral-Nemo-Instruct-2407` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| FAISS | `IndexFlatL2` |
| Chunking | 50 words |
| Overlap | 5 words |
| Retrieval | Top 3 |
| Generation | Sampling |
| Grounding instruction | No |
| Missing-answer fallback | No |
| Retrieved context passed to LLM | First retrieved chunk |
| Retrieval scores | Not exposed |
| Answer-only decoding | No |

## Main behavior

The baseline successfully demonstrates the basic RAG architecture.

However, the executed experiment showed unsupported generation.

Examples included:

- The PDF said the main campus was in **the capital city** and covered **over 300 acres**, while the generated answer added **Baghdad, Iraq** and **over 500 acres**.
- The online-program question produced unsupported claims about online courses and related programs.
- The financial-aid question introduced claims that were not present in the PDF.
- The instruction-language question produced **English and Arabic**, although the PDF did not explicitly state those as teaching languages.

## Main lesson

```text
Relevant retrieval
      ↓
LLM
      ↓
Not necessarily grounded answer
```

This notebook is therefore best treated as the **baseline/reference implementation**.

---

# 6. Notebook 2 — Improved English RAG

File:

```text
Tips-Hindawi-English-Rag-Task-1_.ipynb
```

README:

```text
README_Tips-Hindawi-English-Rag-Task-1.md
```

Branch:

```text
english-improved
```

## Main configuration

| Component | Improved English |
|---|---|
| Generator | `Qwen/Qwen2.5-3B-Instruct` |
| Embeddings | `intfloat/multilingual-e5-base` |
| FAISS | `IndexFlatIP` |
| Embedding normalization | Yes |
| E5 `passage:` prefix | Yes |
| E5 `query:` prefix | Yes |
| Chunking | 100 words |
| Overlap | 20 words |
| Retrieval | Top 3 |
| Generation | Deterministic |
| Grounding instruction | Yes |
| Missing-answer fallback | Yes |
| Retrieved context passed to LLM | All retrieved chunks |
| Retrieval scores | Yes |
| Answer-only decoding | Yes |

## Executed dataset statistics

The notebook reported:

```text
Characters: 4112
Chunks: 8
Embedding dimension: 768
```

## Main improvement

The generator is explicitly instructed to:

- use only retrieved context,
- avoid inventing facts,
- preserve important factual details,
- answer in the same language,
- report when the requested information is not specified.

## Observed behavior

For information supported by the PDF, the answer was grounded.

For unsupported questions, the system returned:

```text
The information is not specified in the document.
```

This was particularly useful for questions about:

- online programs,
- financial aid specifically for international students,
- instruction languages.

## Main lesson

The improved version shows that stronger prompting, better embeddings, normalized retrieval, and complete retrieved context can substantially improve grounding behavior compared with the simple baseline.

---

# 7. Notebook 3 — Arabic RAG

File:

```text
Tips-Hindawi-Arabic-Rag-Task_.ipynb
```

README:

```text
README_Tips-Hindawi-Arabic-Rag-Task.md
```

Branch:

```text
arabic-rag
```

## Main configuration

| Component | Arabic |
|---|---|
| Generator | `Qwen/Qwen2.5-3B-Instruct` |
| Embeddings | `intfloat/multilingual-e5-base` |
| FAISS | `IndexFlatIP` |
| Embedding normalization | Yes |
| E5 `passage:` prefix | Yes |
| E5 `query:` prefix | Yes |
| Chunking | 100 words |
| Overlap | 20 words |
| Retrieval | Top 5 |
| Generation | Deterministic |
| Same-language instruction | Yes |
| Retrieval scores | Yes |
| Evidence inspection | Yes |
| PDF | Arabic university information |

## Executed dataset statistics

The notebook reported:

```text
Characters: 4928
Chunks: 10
Embedding dimension: 768
```

## Arabic questions

The notebook tested:

```text
أين تقع جامعة هنداوي؟

هل تقدم الجامعة برامج دراسية عبر الإنترنت؟

هل توجد مساعدات مالية للطلاب الدوليين؟

ما هي اللغات المستخدمة في التدريس؟
```

## Observed behavior

### Location

The model returned an answer supported by the document:

- capital city,
- more than 300 acres,
- 12 academic buildings,
- central library with more than one million volumes,
- four student residence buildings,
- medical center,
- sports complex and stadium,
- innovation and entrepreneurship center.

The top similarity score was approximately:

```text
0.8760
```

### Online programs

The model answered that the university offers online programs because the document mentions online alumni seminars.

This is an **unsupported inference**.

The document's mention of online alumni seminars does not itself establish online academic programs.

### Financial aid for international students

The model answered that international students receive need-based financial aid and merit scholarships.

The document mentions:

- need-based grants,
- merit scholarships,
- an Office of International Programs that assists international students.

But it does not explicitly state that the listed financial aid is specifically available to international students.

This is another **unsupported inference**.

### Instruction languages

The model answered that the information is not directly specified.

This is consistent with the document evidence.

---

# 8. English Baseline vs Improved English

The main technical progression is:

| Feature | English Baseline | Improved English |
|---|---|---|
| Generator | Mistral-Nemo-Instruct-2407 | Qwen2.5-3B-Instruct |
| Embeddings | all-MiniLM-L6-v2 | multilingual-e5-base |
| FAISS | IndexFlatL2 | IndexFlatIP |
| Normalization | No | Yes |
| E5 prefixes | No | Yes |
| Chunk size | 50 | 100 |
| Overlap | 5 | 20 |
| Top K | 3 | 3 |
| Context to LLM | First chunk only | All retrieved chunks |
| Grounding instruction | No | Yes |
| Missing-info fallback | No | Yes |
| Same-language rule | No | Yes |
| Retrieval scores | No | Yes |
| Answer-only decoding | No | Yes |
| Generation | Sampling | Deterministic |

The key progression is:

```text
Baseline
   ↓
Simple retrieval + simple prompt
   ↓
Unsupported generation

Improved
   ↓
Better embeddings
   +
Better retrieval configuration
   +
All retrieved context
   +
Grounding instructions
   +
Missing-information fallback
   +
Deterministic generation
   ↓
More controlled answers
```

---

# 9. Improved English vs Arabic

The English improved and Arabic notebooks share the same core architecture:

```text
Qwen2.5-3B-Instruct
        +
multilingual-e5-base
        +
FAISS IndexFlatIP
        +
Normalized embeddings
        +
E5 query/passage prefixes
        +
Deterministic generation
```

The main differences are:

| Feature | Improved English | Arabic |
|---|---|---|
| Knowledge source | English PDF | Arabic PDF |
| Questions | English | Arabic |
| Chunk size | 100 | 100 |
| Overlap | 20 | 20 |
| Top K | 3 | 5 |
| Max new tokens | 250 | 350 |
| Characters | 4112 | 4928 |
| Chunks | 8 | 10 |
| Embedding dimension | 768 | 768 |

The architecture is therefore largely reusable across languages.

---

# 10. What the Experiments Demonstrate

The three notebooks demonstrate three stages of RAG development.

## Stage 1 — Basic RAG

The baseline shows how to:

- read a PDF,
- split text,
- create embeddings,
- build FAISS,
- retrieve chunks,
- prompt an LLM.

## Stage 2 — Controlled RAG

The improved English version adds:

- multilingual embeddings,
- E5 query/passage formatting,
- normalized embeddings,
- inner-product retrieval,
- complete retrieved context,
- explicit grounding instructions,
- missing-information fallback,
- deterministic generation,
- evidence inspection.

## Stage 3 — Cross-language RAG

The Arabic notebook demonstrates that the same architecture can be reused for an Arabic knowledge source and Arabic questions.

---

# 11. Important RAG Lessons

## Lesson 1 — Retrieval quality matters

If the retriever returns unrelated or only partially relevant information, the generator has a weak evidence base.

## Lesson 2 — Retrieval alone is not enough

A relevant chunk does not automatically force the LLM to stay within the evidence.

## Lesson 3 — Prompt grounding helps

Explicit instructions such as:

```text
Answer ONLY using the information provided in the retrieved context.
Do not invent or add information.
```

can make the model more controlled.

## Lesson 4 — Missing information should be handled explicitly

A RAG system should distinguish between:

```text
The document contains the answer.
```

and:

```text
The document does not provide the answer.
```

## Lesson 5 — Evidence inspection is essential

A final answer should be evaluated together with:

```text
Retrieved chunk
+
Similarity score
+
Generated answer
```

## Lesson 6 — Multilingual RAG needs multilingual retrieval

For Arabic and English, a multilingual embedding model provides a common retrieval representation.

## Lesson 7 — Even improved RAG can fail

The Arabic experiment is especially useful because it shows that a system can still make unsupported inferences even when:

- the prompt says not to invent,
- the embeddings are multilingual,
- the vectors are normalized,
- the retrieved context is printed,
- generation is deterministic.

This is why production RAG systems often add stronger verification mechanisms.

---

# 12. Current Limitations

These notebooks are educational experiments rather than production RAG systems.

Common limitations include:

- word-based chunking,
- small document size,
- no reranker,
- no claim-level verification,
- no citation-generation layer,
- local in-memory FAISS index,
- dependency/version sensitivity,
- PDF extraction artifacts,
- reliance on a single LLM generation step.

---

# 13. Recommended Repository Workflow

The planned Git workflow is:

```text
main
 │
 ├── english-baseline
 │     ├── Tips-Hindawi-English-Rag-Task_.ipynb
 │     └── README_Tips-Hindawi-English-Rag-Task.md
 │
 ├── english-improved
 │     ├── Tips-Hindawi-English-Rag-Task-1_.ipynb
 │     └── README_Tips-Hindawi-English-Rag-Task-1.md
 │
 └── arabic-rag
       ├── Tips-Hindawi-Arabic-Rag-Task_.ipynb
       └── README_Tips-Hindawi-Arabic-Rag-Task.md
```

The `main` branch contains:

```text
README.md
```

with the overall project explanation.

---

# 14. GitHub Branch Naming

Use:

```text
main
english-baseline
english-improved
arabic-rag
```

These names are descriptive and easy to understand.

---

# 15. Final Project Summary

The project evolves from a simple lecture-style RAG system into a more controlled multilingual RAG architecture.

The overall progression is:

```text
Lecture RAG
    ↓
English Baseline
    ↓
Improved English RAG
    ↓
Arabic RAG
```

The final architecture used by the improved multilingual experiments is:

```text
PDF
 ↓
PyPDF2
 ↓
100-word chunks
 ↓
multilingual-e5-base
 ↓
Normalized 768-D vectors
 ↓
FAISS IndexFlatIP
 ↓
Top-K retrieval
 ↓
Qwen2.5-3B-Instruct
 ↓
Grounded answer
```

The experiments demonstrate both the usefulness and the limitations of RAG:

- RAG can retrieve relevant document evidence.
- Multilingual embeddings allow the architecture to support Arabic and English.
- Better prompts can reduce unsupported answers.
- Deterministic generation improves reproducibility.
- Evidence inspection makes retrieval behavior easier to analyze.
- However, an LLM can still make unsupported inferences even when the prompt says not to.

The next logical improvement would be an explicit **answer verification / claim-to-evidence checking layer** before returning the final response.
