# Tips Hindawi University --- English RAG Task 1

## Overview

This notebook implements a **Simple Retrieval-Augmented Generation
(RAG)** system for answering questions about the
`Tips Hindawi University Info.pdf` document.

The implementation follows the same overall architecture introduced in
the RAG lecture:

``` text
PDF
 ↓
Text Extraction
 ↓
Text Chunking
 ↓
Embedding Generation
 ↓
FAISS Vector Index
 ↓
Question Embedding
 ↓
Relevant Context Retrieval
 ↓
Grounded Prompt
 ↓
Instruction-Tuned LLM
 ↓
Final Answer
```

This version was designed as a more robust/bilingual implementation than
the original lecture-style baseline. It uses:

-   **Generator:** `Qwen/Qwen2.5-3B-Instruct`
-   **Embedding model:** `intfloat/multilingual-e5-base`
-   **Vector database/index:** FAISS `IndexFlatIP`
-   **PDF extraction:** PyPDF2
-   **Execution environment:** Kaggle Notebook

The generator and embedding model were intentionally selected so the
same RAG architecture can later be reused with Arabic questions and an
Arabic version of the document.

------------------------------------------------------------------------

## 1. Objective

The goal of the notebook is to build a small document-grounded QA system
that:

1.  Reads the Tips Hindawi University PDF.
2.  Splits the document into smaller text chunks.
3.  Converts the chunks into semantic vector embeddings.
4.  Stores the vectors in a FAISS index.
5.  Retrieves the most relevant chunks for a user question.
6.  Provides only the retrieved context to the language model.
7.  Instructs the model not to invent information.
8.  Returns an answer in the same language as the question.

A key design objective is **grounding**: if the requested information is
not supported by the retrieved document context, the model is instructed
to return:

> `The information is not specified in the document.`

------------------------------------------------------------------------

## 2. Models and Technologies

### 2.1 Generator --- Qwen2.5-3B-Instruct

The notebook uses:

``` python
model_name = "Qwen/Qwen2.5-3B-Instruct"
```

The model is loaded with:

``` python
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
```

The model is used as the final answer generator after relevant document
chunks have been retrieved.

The implementation uses the Qwen chat template and a system instruction
that explicitly tells the model to answer only from the supplied
context.

------------------------------------------------------------------------

### 2.2 Embedding Model --- multilingual-e5-base

The embedding model is:

``` python
intfloat/multilingual-e5-base
```

The notebook uses the E5 prefixes:

``` text
passage:
```

for document chunks and:

``` text
query:
```

for user questions.

Embeddings are normalized:

``` python
normalize_embeddings=True
```

This allows cosine similarity to be represented using inner product in
FAISS.

This multilingual embedding choice is also important for the planned
Arabic version of the task.

------------------------------------------------------------------------

### 2.3 FAISS

The notebook creates:

``` python
faiss.IndexFlatIP(dim)
```

Because the embeddings are normalized, inner product corresponds to
cosine similarity.

The index stores all document chunk embeddings and retrieves the top `k`
most relevant chunks for a question.

------------------------------------------------------------------------

## 3. Notebook Pipeline

### Step 1 --- Install dependencies

The notebook installs:

``` python
transformers==4.52.4
sentence-transformers
PyPDF2
faiss-cpu
accelerate
```

These packages provide the language model, embedding model, PDF
extraction, vector search, and model-loading support.

------------------------------------------------------------------------

### Step 2 --- Import libraries

The notebook imports:

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import torch
import faiss
import os
```

------------------------------------------------------------------------

### Step 3 --- Load the generator

The Qwen model and tokenizer are loaded using Hugging Face Transformers.

`device_map="auto"` allows Transformers to automatically place the model
on the available hardware.

------------------------------------------------------------------------

### Step 4 --- Define controlled generation

The `generate_text()` function uses a system message containing the
grounding rule:

``` text
Answer ONLY from the supplied context.
If the context does not contain the answer, say that the information is not specified in the document.
Never invent facts.
```

Generation is deterministic:

``` python
do_sample=False
temperature=0.0
```

The function also removes the original prompt from the returned sequence
and decodes only newly generated tokens.

This is an important improvement over the baseline notebook, where the
generated output includes the original prompt.

------------------------------------------------------------------------

## 4. PDF Processing

### 4.1 Text extraction

The function:

``` python
extract_text_from_pdf(pdf_path)
```

reads every page using `PdfReader` and combines the extracted text into
one string.

------------------------------------------------------------------------

### 4.2 Chunking

The notebook uses:

``` python
chunk_size=100
overlap=20
```

The PDF text is split into word-based chunks.

The overlap helps preserve context between neighboring chunks.

For example:

``` text
Chunk A
[words 1 ... 100]

Chunk B
[words 81 ... 180]
```

------------------------------------------------------------------------

## 5. Embedding and Index Construction

The extracted chunks are converted into vectors using
`multilingual-e5-base`.

For this executed run, the notebook reported:

``` text
Number of characters: 4112
Number of chunks: 8
Embedding dimension: 768
```

Therefore, the processed English document produced:

-   **4,112 extracted characters**
-   **8 text chunks**
-   **768-dimensional embeddings**

The resulting vectors were stored in a FAISS inner-product index.

------------------------------------------------------------------------

## 6. Retrieval

The `search_index()` function embeds the user question as:

``` text
query: <question>
```

and searches the FAISS index.

The notebook retrieves:

``` python
k=3
```

chunks for each question.

Each result contains:

``` python
{
    "chunk": ...,
    "score": ...
}
```

The notebook also includes an evidence-inspection section that prints
the retrieved chunks and similarity scores.

------------------------------------------------------------------------

## 7. RAG Answer Generation

The main function is:

``` python
answer_question(question, k=3, max_new_tokens=250)
```

Its workflow is:

``` text
Question
   ↓
Retrieve top 3 chunks
   ↓
Combine retrieved chunks
   ↓
Build grounded prompt
   ↓
Qwen2.5-3B-Instruct
   ↓
Answer
```

The prompt explicitly instructs the model:

-   Use only retrieved context.
-   Do not add outside knowledge.
-   If the answer is not explicitly supported, state that it is not
    specified.
-   Answer in the same language as the question.

------------------------------------------------------------------------

# 8. Required Lab Questions and Observed Results

The notebook evaluates four questions.

## Question 1

**Question:**

``` text
Where is Tips Hindawi University located?
```

**Observed answer:**

``` text
The main campus of Tips Hindawi University is located in the capital city and spans over 300 acres.
```

This answer is grounded in the PDF. Importantly, the PDF says **"the
capital city"** but does not name the city.

The retrieved evidence included a highly relevant chunk with a score of
approximately:

``` text
0.8425
```

------------------------------------------------------------------------

## Question 2

**Question:**

``` text
Does the university offer online programs?
```

**Observed answer:**

``` text
The information is not specified in the document.
```

This is an important RAG behavior because the PDF does not explicitly
state that online programs are offered.

Instead of inventing an answer, the model followed the grounding
instruction.

------------------------------------------------------------------------

## Question 3

**Question:**

``` text
Is there financial aid for international students?
```

**Observed answer:**

``` text
The information is not specified in the document.
```

The document does mention scholarships, need-based grants, research
fellowships, and support for international students through the Office
of International Programs. However, it does not explicitly establish
that financial aid is specifically available to international students.

The notebook therefore avoids making an unsupported connection.

------------------------------------------------------------------------

## Question 4

**Question:**

``` text
What languages are used for instruction?
```

**Observed answer:**

``` text
The information is not specified in the document.
```

The PDF mentions an English proficiency requirement for admission, but
it does not explicitly state the language or languages of instruction.

The RAG system correctly avoids treating the English proficiency
requirement as proof that all instruction is in English.

------------------------------------------------------------------------

# 9. Retrieval Evidence

The notebook includes a dedicated evidence-inspection cell.

For example, for the location question, the retrieved chunks included:

``` text
... Tips Hindawi University (THU) is a premier institution ...
...
The main campus is located in the capital city and spans over 300 acres.
```

with the highest observed retrieval score around:

``` text
0.8425
```

This provides a useful way to inspect not only the generated answer but
also the evidence supplied to the model.

------------------------------------------------------------------------

# 10. Important Runtime Note

The executed notebook displayed a Transformers warning that some
generation flags were not valid or were ignored:

``` text
The following generation flags are not valid and may be ignored:
['temperature', 'top_p', 'top_k']
```

The notebook still produced answers, but this warning indicates that the
generation configuration should be kept consistent with the selected
deterministic generation settings.

The model-loading/generation environment also depends on GPU
availability.

------------------------------------------------------------------------

# 11. Why This Version Is More Grounded

Compared with a simple prompt + retrieved chunk implementation, this
notebook adds several safeguards:

### 1. Explicit system instruction

The model is directly instructed not to invent facts.

### 2. Retrieved-context-only prompt

The answer-generation prompt contains the retrieved document evidence.

### 3. Missing-information fallback

The model is given a specific response when the document does not
contain the requested information.

### 4. Same-language response rule

The model is instructed to answer in the same language as the question.

### 5. Better retrieval configuration

The notebook uses:

``` text
multilingual-e5-base
```

with E5's `passage:` and `query:` conventions.

### 6. Normalized embeddings + inner product

This provides cosine-similarity-style retrieval through FAISS.

### 7. Evidence inspection

The notebook prints the retrieved chunks and scores so the RAG behavior
can be examined rather than evaluating only the final generated text.

------------------------------------------------------------------------

# 12. Reproducibility

To reproduce the notebook:

1.  Open the notebook in Kaggle.
2.  Make sure the Tips Hindawi University PDF is available at the
    expected input path.
3.  Run the dependency installation cell.
4.  Load Qwen2.5-3B-Instruct.
5.  Run the PDF extraction and chunking cells.
6.  Generate multilingual embeddings.
7.  Build the FAISS index.
8.  Run the four required questions.
9.  Inspect both answers and retrieved evidence.

The PDF path used in the executed notebook was:

``` text
/kaggle/input/datasets/andrewsameh20/tips-hindawi-university-information/Tips Hindawi University Info.pdf
```

------------------------------------------------------------------------

## 13. Summary

This notebook implements an English document-based RAG system using:

``` text
Qwen2.5-3B-Instruct
        +
multilingual-e5-base
        +
FAISS
        +
PyPDF2
```

The executed results demonstrate a key RAG property: when the document
contains an answer, the system can return a grounded response; when the
requested information is not supported by the document, the system can
explicitly report that the information is not specified instead of
relying on outside knowledge.

The notebook is also structured so that the same architecture can later
be reused for the Arabic version of the Tips Hindawi University
document.
