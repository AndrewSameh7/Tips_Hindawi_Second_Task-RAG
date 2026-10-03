# Tips Hindawi University — Arabic RAG Task

## Overview

This notebook implements an **Arabic document-based Retrieval-Augmented Generation (RAG)** system for answering questions about the Arabic version of the Tips Hindawi University information document.

The notebook keeps the same overall architecture used in the lecture and in the improved English RAG implementation:

```text
Arabic PDF
   ↓
Text Extraction
   ↓
Text Chunking
   ↓
Multilingual Embeddings
   ↓
FAISS Vector Index
   ↓
Arabic Question
   ↓
Relevant Context Retrieval
   ↓
Grounded Prompt
   ↓
Qwen2.5-3B-Instruct
   ↓
Arabic Answer
```

The implementation is designed as a **bilingual-capable pipeline**: the generator and embedding model are selected to work with both English and Arabic, while this experiment uses the Arabic university-information PDF as the knowledge source.

---

# 1. Objective

The main objectives of the notebook are to:

1. Load an instruction-tuned language model.
2. Read the Arabic university-information PDF.
3. Extract its text.
4. Split the text into overlapping chunks.
5. Convert the chunks into multilingual semantic embeddings.
6. Store the embeddings in a FAISS vector index.
7. Convert Arabic questions into query embeddings.
8. Retrieve the most relevant document chunks.
9. Provide the retrieved context to the LLM.
10. Generate answers in the same language as the question.
11. Inspect the retrieved evidence and similarity scores.

A major goal is **grounded question answering**, meaning that the model is instructed to use the retrieved document context rather than inventing external information.

---

# 2. Models and Technologies

## 2.1 Generator — Qwen2.5-3B-Instruct

The notebook uses:

```python
model_name = "Qwen/Qwen2.5-3B-Instruct"
```

The tokenizer and model are loaded with Hugging Face Transformers:

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
```

`device_map="auto"` allows Transformers to place the model on the available hardware.

Qwen is used as the final answer generator after the retrieval stage.

---

## 2.2 Embedding Model — multilingual-e5-base

The embedding model is:

```python
intfloat/multilingual-e5-base
```

The notebook uses E5-style prefixes:

```text
passage:
```

for document chunks and:

```text
query:
```

for user questions.

Embeddings are normalized:

```python
normalize_embeddings=True
```

This makes inner product in FAISS equivalent to cosine-similarity-style retrieval.

This model is particularly suitable for the notebook's bilingual design because the same embedding model is used for Arabic and English text.

---

## 2.3 FAISS

The notebook creates:

```python
faiss.IndexFlatIP(dim)
```

The index uses **Inner Product (IP)** similarity.

Because the embeddings are normalized, this is used as cosine-similarity-style retrieval.

The search function returns:

- rank,
- chunk ID,
- retrieved chunk,
- similarity score.

This makes it possible to inspect not only the final answer but also the evidence retrieved for each question.

---

## 2.4 PDF Processing

The notebook uses:

```python
from PyPDF2 import PdfReader
```

The PDF is read page by page and the extracted text is concatenated into a single string.

---

# 3. Dependencies

The notebook installs:

```python
!pip install -q transformers==4.52.4 sentence-transformers PyPDF2 faiss-cpu accelerate
```

Main libraries:

| Library | Purpose |
|---|---|
| `transformers` | Load and run Qwen2.5-3B-Instruct |
| `sentence-transformers` | Generate multilingual embeddings |
| `PyPDF2` | Extract text from the PDF |
| `faiss-cpu` | Vector similarity search |
| `accelerate` | Support model loading/device placement |
| `torch` | Tensor operations and model inference |

---

# 4. Generation Function

The main generation function is:

```python
def generate_text(prompt, max_new_tokens=350):
```

It creates a Qwen chat conversation containing a system instruction and the user prompt.

The system instruction tells the model to:

- answer only from the retrieved context,
- answer in the same language as the question,
- provide a complete answer,
- include all relevant facts,
- preserve numbers, measurements, locations, names, and dates,
- avoid inventing or adding information.

The generation is deterministic:

```python
do_sample=False
```

The notebook also removes the original input tokens before decoding:

```python
generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
```

This means the returned text is intended to contain only the generated answer.

---

# 5. PDF Text Extraction

The function:

```python
extract_text_from_pdf(pdf_path)
```

uses `PdfReader` and loops through all pages:

```python
for page in reader.pages:
    page_text = page.extract_text() or ""
    full_text += page_text + "\n"
```

This produces one combined text string for chunking.

---

# 6. Chunking Strategy

The general chunking function is defined as:

```python
def chunk_text(text, chunk_size=150, overlap=40):
```

However, the actual experiment calls it with:

```python
chunks = chunk_text(
    text,
    chunk_size=100,
    overlap=20
)
```

Therefore, the executed experiment uses:

- **100 words per chunk**
- **20 words of overlap**

The overlap helps preserve information across chunk boundaries.

The chunking is word-based rather than section-aware or semantic.

---

# 7. Embedding Generation

Each document chunk is converted into:

```text
passage: <chunk>
```

before being passed to `multilingual-e5-base`.

The notebook uses:

```python
embedding_model.encode(
    passage_texts,
    convert_to_numpy=True,
    normalize_embeddings=True
)
```

The executed run produced:

```text
Number of characters: 4928
Number of chunks: 10
Embedding dimension: 768
```

Therefore, this Arabic experiment produced:

- **4,928 extracted characters**
- **10 text chunks**
- **768-dimensional embeddings**

---

# 8. FAISS Index Construction

The embeddings are stored in:

```python
faiss.IndexFlatIP(dim)
```

The index is populated with:

```python
index.add(embeddings)
```

The system therefore has the following retrieval structure:

```text
Arabic PDF
   ↓
10 chunks
   ↓
768-dimensional normalized vectors
   ↓
FAISS IndexFlatIP
```

---

# 9. Retrieval

The search function is:

```python
search_index(query, model, index, chunks, k=5)
```

The Arabic question is converted to:

```text
query: <Arabic question>
```

and encoded using the same multilingual embedding model.

The FAISS search returns the top `k` results.

The returned result structure contains:

```python
{
    "rank": rank,
    "chunk_id": int(idx),
    "chunk": chunks[idx],
    "score": float(score)
}
```

The notebook therefore exposes retrieval scores for inspection.

---

# 10. Answer Generation Pipeline

The notebook defines:

```python
answer_question(question, k=5, max_new_tokens=350)
```

The workflow is:

```text
Arabic Question
      ↓
Query Embedding
      ↓
FAISS Retrieval
      ↓
Top 5 Chunks
      ↓
Combine Retrieved Context
      ↓
Grounded Prompt
      ↓
Qwen2.5-3B-Instruct
      ↓
Arabic Answer
```

The function returns:

```python
answer, top_chunks, prompt
```

This is useful because the experiment can inspect both:

1. the final generated answer,
2. the evidence that was retrieved.

---

# 11. Arabic Questions Tested

The notebook evaluates four Arabic questions:

```text
أين تقع جامعة هنداوي؟

هل تقدم الجامعة برامج دراسية عبر الإنترنت؟

هل توجد مساعدات مالية للطلاب الدوليين؟

ما هي اللغات المستخدمة في التدريس؟
```

---

# 12. Executed Results

## Question 1 — University Location

**Question:**

```text
أين تقع جامعة هنداوي؟
```

**Observed answer:**

```text
جامعة هنداوي تقع في العاصمة. الحرم الجامعي الرئيسي يمتد على مساحة تزيد عن 300 فدان. يضم: 12 مبنى أكاديميًا، مكتبة مركزية تحتوي على أكثر من مليون مجلد، أربعة مبانٍ سكنية للطالب، مركزًا طبيًا، مجمعًا رياضيًا واستادًا، ومركزًا للابتكار وريادة الأعمال.
```

The top retrieved result had a similarity score of approximately:

```text
0.8760
```

The retrieved evidence explicitly states that the main campus is in the capital and spans more than 300 acres.

This result is well supported by the retrieved document context.

---

## Question 2 — Online Programs

**Question:**

```text
هل تقدم الجامعة برامج دراسية عبر الإنترنت؟
```

**Observed answer:**

```text
نعم، تقدم الجامعة برامج دراسية عبر الإنترنت. حيث أن هناك إشارة واضحة إلى تنظيم شبكة خريجي الجامعة ندوات عبر الإنترنت كل ثالثة أشهر، وهذا يشير إلى تقديم الجامعة برامج دراسية عبر الإنترنت.
```

The top retrieved chunk had a similarity score of approximately:

```text
0.8300
```

### Important grounding observation

The retrieved text says that the **THU alumni network organizes online seminars every three months**.

That statement is not the same as saying that the university offers online academic programs.

Therefore, the generated conclusion:

```text
نعم، تقدم الجامعة برامج دراسية عبر الإنترنت.
```

is **not fully supported by the retrieved evidence**.

This is an important example of a RAG failure mode: the retriever found a semantically related phrase, but the generator made a stronger claim than the document supports.

---

## Question 3 — Financial Aid for International Students

**Question:**

```text
هل توجد مساعدات مالية للطلاب الدوليين؟
```

**Observed answer:**

```text
نعم، يوجد مساعدات مالية للطلاب الدوليين. يتم تقديم منح قائمة على الحاجة المالية للطلاب الدوليين الذين يحتاجون إلى مساعدة مالية. كما يتم تقديم منح التفوق التي تغطي من 25% إلى 100% من الرسوم الدراسية.
```

The top retrieved result had a similarity score of approximately:

```text
0.7916
```

The retrieved context contains:

- need-based grants,
- merit scholarships covering 25%–100% of tuition,
- an Office of International Programs that assists international students.

However, the document does **not explicitly establish** that the listed financial aid is specifically available to international students.

Therefore, the generated answer makes a stronger connection than the retrieved evidence directly supports.

This is another example of **unsupported inference / hallucination** despite the grounding instruction.

---

## Question 4 — Instruction Languages

**Question:**

```text
ما هي اللغات المستخدمة في التدريس؟
```

**Observed answer:**

```text
المعلومات المذكورة في المستند لا تشير إلى اللغات المستخدمة في التدريس بشكل مباشر. لذا، المعلومة غير مذكورة في المستند.
```

The answer correctly avoids treating the English proficiency requirement as proof of the language of instruction.

The PDF mentions English admission requirements such as IELTS and TOEFL, but it does not explicitly state the language or languages used for teaching.

This result is therefore consistent with the available document evidence.

---

# 13. Retrieval Evidence

The notebook includes a dedicated inspection loop that prints:

```text
Rank
Chunk ID
Similarity Score
Retrieved Chunk
```

For example, the location question retrieved:

```text
Rank: 1
Chunk ID: 0
Similarity Score: 0.8760
```

and the retrieved chunk contained the main-campus information.

This evidence-inspection step is important because a RAG system should not be evaluated only by looking at the final generated answer.

---

# 14. Runtime Warning

During execution, Transformers displayed:

```text
The following generation flags are not valid and may be ignored:
['temperature', 'top_p', 'top_k']
```

The notebook itself uses:

```python
do_sample=False
```

and does not explicitly set those sampling parameters in the shown generation call.

The warning should therefore be treated as an environment/model-generation configuration note rather than evidence that sampling was intentionally enabled in this notebook.

---

# 15. Important Strengths

This Arabic implementation improves on the simple lecture-style baseline in several ways.

### 1. Arabic document support

The knowledge source is an Arabic PDF.

### 2. Arabic question answering

The tested questions are written in Arabic and the model returns Arabic answers.

### 3. Multilingual embeddings

`multilingual-e5-base` is used instead of an English-focused MiniLM embedding setup.

### 4. E5 query/passage formatting

The notebook uses:

```text
passage:
```

for stored chunks and:

```text
query:
```

for questions.

### 5. Normalized embeddings

Embeddings use:

```python
normalize_embeddings=True
```

### 6. Inner-product retrieval

FAISS uses:

```python
IndexFlatIP
```

### 7. Deterministic generation

The notebook uses:

```python
do_sample=False
```

which makes the generation configuration more reproducible.

### 8. Evidence inspection

The notebook prints retrieved chunks and similarity scores.

### 9. Same-language instruction

The system prompt explicitly requests that the answer use the same language as the question.

---

# 16. Limitations

Despite the improvements, the executed results show important limitations.

## 16.1 Retrieval can be semantically related but insufficient

For the online-program question, the retriever found text about **online alumni seminars**, and the model incorrectly generalized this into online academic programs.

## 16.2 Grounding instructions do not guarantee factual grounding

The system prompt explicitly says not to invent information, but the executed results still contain unsupported conclusions.

Therefore:

```text
Grounding Prompt
      +
Good Retrieval
      ≠
Guaranteed Factual Answer
```

## 16.3 No explicit missing-information fallback rule

Unlike the improved English README's documented fallback design, this Arabic notebook's system prompt says not to invent information but does not explicitly define a fixed fallback phrase such as:

```text
The information is not specified in the document.
```

The model nevertheless produced an appropriate missing-information response for the instruction-language question.

## 16.4 Word-based chunking

The PDF is split by word count rather than by:

- section,
- paragraph,
- heading,
- semantic unit.

This can separate related information across chunks.

## 16.5 Small document

Only 10 chunks were created, so this is an educational/small-scale RAG experiment rather than a production retrieval system.

## 16.6 No reranker

There is no cross-encoder or reranking stage after FAISS retrieval.

## 16.7 No answer verification stage

The system does not perform a separate verification step that checks whether every claim in the generated answer is directly supported by retrieved evidence.

---

# 17. Reproducibility

The notebook was executed in a Kaggle environment.

The recorded PDF path in the notebook is:

```text
/kaggle/input/datasets/andrewsameh20/tips-hindawi-university-arabic-information/.pdf
```

### Reproduction steps

1. Open the notebook in Kaggle.
2. Attach the Arabic Tips Hindawi University PDF.
3. Verify the PDF path.
4. Install the required packages.
5. Load Qwen2.5-3B-Instruct.
6. Extract the Arabic PDF text.
7. Create 100-word chunks with 20-word overlap.
8. Generate `multilingual-e5-base` embeddings.
9. Normalize the embeddings.
10. Build `FAISS IndexFlatIP`.
11. Run the four Arabic questions.
12. Inspect the retrieved chunks and similarity scores.
13. Compare the generated answers with the actual retrieved evidence.

---

# 18. Overall Result

The Arabic notebook successfully demonstrates the complete Arabic RAG pipeline:

```text
Arabic PDF
   ↓
PDF Extraction
   ↓
100-word Chunks / 20-word Overlap
   ↓
multilingual-e5-base
   ↓
768-D Normalized Embeddings
   ↓
FAISS IndexFlatIP
   ↓
Top-5 Retrieval
   ↓
Qwen2.5-3B-Instruct
   ↓
Arabic Answer
```

The experiment also demonstrates an important practical lesson:

> A RAG system can retrieve relevant information and still generate a conclusion that is stronger than the evidence.

The location and instruction-language questions produced answers that were consistent with the document, while the online-program and international-financial-aid questions showed unsupported inference.

Therefore, retrieval quality and prompt grounding are important, but a stronger production system would also need answer verification or claim-level evidence checking.

---

# 19. Summary

This notebook is the **Arabic RAG implementation** of the Tips Hindawi University lab.

It uses:

```text
Qwen2.5-3B-Instruct
        +
multilingual-e5-base
        +
FAISS IndexFlatIP
        +
PyPDF2
        +
Arabic University PDF
```

The main technical characteristics are:

| Component | Arabic Notebook |
|---|---|
| Generator | Qwen2.5-3B-Instruct |
| Embeddings | multilingual-e5-base |
| Vector Index | FAISS IndexFlatIP |
| Similarity | Inner Product / cosine-style after normalization |
| Chunk Size | 100 words |
| Chunk Overlap | 20 words |
| Retrieved Chunks | Top 5 |
| Embedding Dimension | 768 |
| Extracted Characters | 4,928 |
| Generation | Deterministic (`do_sample=False`) |
| Question Language | Arabic |
| Evidence Inspection | Yes |
| Reranker | No |
| Claim Verification | No |

The notebook demonstrates that the same RAG architecture can be adapted from English to Arabic while keeping the core retrieval and generation design consistent.
