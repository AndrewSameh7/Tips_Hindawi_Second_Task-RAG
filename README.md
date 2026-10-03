# Tips Hindawi University --- English RAG Baseline

## Overview

This notebook contains the earlier, lecture-style implementation of a
**Simple Retrieval-Augmented Generation (RAG)** system for the
`Tips Hindawi University Info.pdf` document.

It follows the basic RAG pipeline:

``` text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS Search
 ↓
Retrieved Chunk
 ↓
Prompt
 ↓
Language Model
 ↓
Answer
```

The implementation uses:

-   **Generator:** `mistralai/Mistral-Nemo-Instruct-2407`
-   **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
-   **Vector index:** FAISS `IndexFlatL2`
-   **PDF extraction:** PyPDF2
-   **Execution environment:** Kaggle Notebook

This notebook is useful as the **baseline implementation** for
understanding the progression from a simple RAG pipeline to the more
controlled RAG implementation in
`Tips-Hindawi-English-Rag-Task-1_.ipynb`.

------------------------------------------------------------------------

# 1. Objective

The notebook demonstrates the fundamental steps required to build a
simple RAG system:

1.  Load an instruction-tuned language model.
2.  Define a text-generation function.
3.  Read the university PDF.
4.  Split the document into chunks.
5.  Convert chunks to embeddings.
6.  Build a FAISS index.
7.  Retrieve relevant chunks.
8.  Insert retrieved text into a prompt.
9.  Generate an answer.

The implementation is intentionally simple and closely follows the
lecture-style approach.

------------------------------------------------------------------------

# 2. Models and Technologies

## 2.1 Generator --- Mistral-Nemo-Instruct-2407

The notebook loads:

``` python
mistralai/Mistral-Nemo-Instruct-2407
```

using:

``` python
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

The model is used for text generation after retrieval.

------------------------------------------------------------------------

## 2.2 Generation Strategy

The generation function is:

``` python
def generate_text(prompt, max_length=100, num_return_sequences=1):
```

It uses sampling:

``` python
do_sample=True
top_k=50
top_p=0.95
temperature=0.7
```

This makes generation probabilistic rather than deterministic.

The function returns decoded generated sequences.

------------------------------------------------------------------------

## 2.3 Embedding Model

The notebook uses:

``` python
sentence-transformers/all-MiniLM-L6-v2
```

Chunks are directly encoded:

``` python
embeddings = model.encode(
    chunks,
    convert_to_numpy=True
)
```

------------------------------------------------------------------------

## 2.4 FAISS Index

The baseline uses:

``` python
faiss.IndexFlatL2(dim)
```

This means retrieval is based on L2 distance.

The question is embedded and compared with the stored document
embeddings.

------------------------------------------------------------------------

# 3. PDF Processing

## 3.1 Text Extraction

The function:

``` python
extract_text_from_pdf(pdf_path)
```

loops through every page and concatenates the extracted text.

------------------------------------------------------------------------

## 3.2 Chunking

The general chunking function is:

``` python
def chunk_text(text, chunk_size=500, overlap=50):
```

However, in the actual executed pipeline it is called with:

``` python
chunks = chunk_text(
    text,
    chunk_size=50,
    overlap=5
)
```

Therefore, the actual experiment uses:

-   **50 words per chunk**
-   **5-word overlap**

This is different from the default values written in the function
definition.

------------------------------------------------------------------------

# 4. Retrieval

The search function is:

``` python
search_index(query, model, index, chunks, k=5)
```

But the notebook's question experiments use:

``` python
k=3
```

The function:

1.  Embeds the question.
2.  Searches the FAISS L2 index.
3.  Returns the retrieved chunks.

------------------------------------------------------------------------

# 5. Initial Retrieval Experiment

The first tested question is:

``` text
Where is Tips Hindawi University located?
```

The retrieved chunks included:

### Chunk 1

``` text
1. General Overview Tips Hindawi University (THU) is a premier institution ...
```

### Chunk 2

``` text
leadership, THU has produced more than 150,000 graduates ...
```

### Chunk 3

``` text
Four residence halls - A medical center - Sports complex and stadium ...
```

This demonstrates that the vector search was able to retrieve text from
the relevant general overview/campus sections.

------------------------------------------------------------------------

# 6. Prompt Construction

After retrieval, the notebook selects only the first retrieved chunk:

``` python
chunk = top_chunks[0]
```

and creates:

``` python
prompt = f"Answer the next question: {question} by reading the following text:{chunk}"
```

The generator is then called using this prompt.

This is a very simple RAG prompt.

------------------------------------------------------------------------

# 7. Observed Results

The executed notebook tested four questions.

## Question 1 --- Location

**Question:**

``` text
Where is Tips Hindawi University located?
```

The retrieved chunks were based on the actual PDF, but the generated
answer introduced information that was not contained in the retrieved
PDF text.

The observed generated response stated that the university was located
in:

``` text
Baghdad, Iraq
```

and additionally described:

-   a campus of over 500 acres,
-   a lush green oasis,
-   proximity to transportation networks.

These details were **not present in the retrieved PDF context shown by
the notebook**.

The source PDF instead states that the main campus is located in **the
capital city** and spans **over 300 acres**.

Therefore, this run demonstrates a grounding problem: the generator
supplied unsupported information instead of limiting its answer to the
retrieved document.

------------------------------------------------------------------------

## Question 2 --- Online Programs

**Question:**

``` text
Does the university oer online programs?
```

The retrieved chunks did not explicitly state that the university offers
online programs.

However, the generated response added information about:

-   online courses,
-   executive education programs,
-   lifelong learning.

It then answered:

``` text
Yes, the university offers online programs.
```

Those claims were not present in the retrieved PDF context shown in the
notebook.

This is another example of **hallucination / unsupported generation**.

------------------------------------------------------------------------

## Question 3 --- Financial Aid for International Students

**Question:**

``` text
Is there financial aid for international students?
```

The retrieved context contained information about:

-   tuition fees,
-   merit scholarships,
-   need-based grants,
-   research fellowships.

However, the generated answer introduced a new section about:

``` text
Financial Aid Available for International Students
```

and claimed:

-   international students are not eligible for federal loans,
-   private loans are an option,
-   some universities offer scholarships specifically for international
    students,
-   international students are eligible for on-campus work-study
    programs.

These statements were not present in the retrieved PDF context.

The notebook therefore demonstrates another case where the language
model went beyond the supplied evidence.

------------------------------------------------------------------------

## Question 4 --- Instruction Languages

**Question:**

``` text
What languages are used for instruction?
```

The retrieved chunks shown by the notebook did not explicitly specify
the instructional language.

Nevertheless, the generated response stated:

``` text
The languages used for instruction at the university are English and Arabic.
```

This information was not supported by the retrieved context.

The PDF does mention an English proficiency requirement for
undergraduate admission, but that is not equivalent to an explicit
statement that English and Arabic are the languages of instruction.

------------------------------------------------------------------------

# 8. Important Technical Observation

The baseline notebook demonstrates why **retrieval alone is not
enough**.

The retrieval stage can return relevant document chunks, but the
language model may still use its own learned knowledge or generate
plausible-sounding information when the prompt does not strongly
constrain it.

The baseline prompt is:

``` text
Answer the next question: ... by reading the following text: ...
```

There is no explicit grounding rule.

Therefore:

``` text
Retrieved Context
       ↓
     LLM
       ↓
Potentially grounded OR unsupported answer
```

The improved notebook changes this behavior by explicitly requiring the
model to answer only from the retrieved context.

------------------------------------------------------------------------

# 9. Runtime Warning

During generation, the notebook produced a warning indicating that the
input tensor was on CPU while the model was on CUDA:

``` text
input_ids is on cpu, whereas your model is on cuda
```

The warning explains that this can result in unexpected behavior or
slower generation and recommends moving the inputs to the model's
device.

The baseline notebook also uses sampled generation, which can increase
variability between runs.

------------------------------------------------------------------------

# 10. Baseline Limitations

This implementation is useful for learning the RAG pipeline, but it has
several limitations.

### 1. No explicit grounding instruction

The generator is not clearly told to use only the retrieved context.

### 2. No fallback for missing information

There is no rule such as:

``` text
The information is not specified in the document.
```

### 3. Only the first retrieved chunk is passed to the generator

Although three chunks are retrieved, the code does:

``` python
chunk = top_chunks[0]
```

and sends only that chunk to the model.

### 4. Prompt appears in the generated output

The decoded result includes the original prompt, followed by the
generated answer.

### 5. Sampled generation

The use of:

``` python
do_sample=True
temperature=0.7
top_k=50
top_p=0.95
```

makes the output less deterministic.

### 6. Basic embedding setup

The model is:

``` text
all-MiniLM-L6-v2
```

and the notebook does not use multilingual E5 query/passage formatting.

### 7. No retrieval-score inspection

The search function returns only the text chunks, making it harder to
inspect retrieval confidence.

### 8. Simple word-based chunking

The document is split by word count rather than using document sections,
paragraphs, or semantic boundaries.

------------------------------------------------------------------------

# 11. What This Notebook Demonstrates

Despite its limitations, this notebook successfully demonstrates the
fundamental RAG architecture:

``` text
Document
   ↓
Chunks
   ↓
Embeddings
   ↓
FAISS
   ↓
Query
   ↓
Relevant Chunks
   ↓
Prompt
   ↓
LLM
   ↓
Generated Answer
```

------------------------------------------------------------------------

# 12. Reproducibility

The notebook was executed in Kaggle.

The PDF path used was:

``` text
/kaggle/input/datasets/andrewsameh20/tips-hindawi-university-info/Tips Hindawi University Info.pdf
```

To reproduce the baseline:

1.  Open the notebook in Kaggle.
2.  Make the PDF available at the expected input path.
3.  Install Transformers and RAG dependencies.
4.  Load Mistral-Nemo-Instruct-2407.
5.  Extract the PDF text.
6.  Create 50-word chunks with 5-word overlap.
7.  Generate MiniLM embeddings.
8.  Build the FAISS L2 index.
9.  Retrieve the top 3 chunks.
10. Pass the first retrieved chunk to the generator.
11. Run the four questions.

------------------------------------------------------------------------

# 13. Summary

This notebook is the **baseline English RAG implementation** for the
Tips Hindawi University task.

It demonstrates the essential RAG components successfully, but its
executed results also expose an important weakness of simple RAG
pipelines: a language model can generate plausible information that is
not actually contained in the retrieved document.

The main lesson from this baseline is:

> Good retrieval does not automatically guarantee a grounded answer.

This notebook should therefore be viewed as the **initial/reference
implementation**.
