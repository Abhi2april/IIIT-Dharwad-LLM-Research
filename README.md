# Curriculum-Based Question Generation

This repository contains two approaches for curriculum-based question generation:
1. **Fine-Tuning Approach**: Using Unsloth's FastLanguageModel for fine-tuning large language models (LLMs) on domain-specific datasets.
2. **Retrieval-Augmented Generation (RAG) Approach**: Using LangChain and ChromaDB to generate questions based on retrieved knowledge from documents.

## 1️⃣ Fine-Tuning Approach

### Overview
The fine-tuning approach utilizes Unsloth's FastLanguageModel to finetune a pre-trained model on a dataset containing curriculum-based questions.

### Setup & Installation
```bash
# Install dependencies
mamba install --force-reinstall aiohttp -y
pip install -U "xformers<0.0.26" --index-url https://download.pytorch.org/whl/
pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"

# Fix dependencies
pip install datasets==2.16.0 fsspec==2023.10.0 gcsfs==2023.10.0
```

### Model Loading & Training
```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
```

The dataset is loaded from Hugging Face:
```python
from datasets import load_dataset

dataset = load_dataset("ABHISHEKSINGH0204/math_new", split="train")
```

The fine-tuning process is implemented using LoRA:
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)
```

## 2️⃣ Retrieval-Augmented Generation (RAG) Approach

### Overview
This approach utilizes LangChain, ChromaDB, and Llama 2 to generate questions by retrieving relevant knowledge from a document source.

### Setup & Installation
```bash
pip install transformers==4.33.0 accelerate==0.22.0 einops==0.6.1 langchain==0.300
pip install bitsandbytes==0.41.1 sentence_transformers==2.2.2 chromadb==0.4.12
pip install rouge-score==0.0.4 nltk==3.8.1 bert-score==0.3.11
```

### Model & Tokenizer Initialization
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = '/kaggle/input/llama-2/pytorch/7b-chat-hf/1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

### Query Pipeline
```python
from transformers import pipeline
query_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

### Document Processing & Embeddings
```python
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

document = PyMuPDFLoader("path/to/pdf").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
all_splits = text_splitter.split_documents(document)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")
```

### Retrieval & Question Generation
```python
from langchain.chains import RetrievalQA
retriever = vectordb.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)
```

### Testing the Pipeline
```python
def test_rag(qa, query):
    print(f"Query: {query}\n")
    result = qa.run(query)
    print("\nResult:", result)

query = "Generate 2 mathematics questions based on the provided curriculum PDF."
test_rag(qa, query)
```

## References
1. [Using LLaMA 2.0, FAISS, and LangChain for QA](https://medium.com/@murtuza753/using-llama-2-0-faiss-and-langchain-for-question-answering-on-your-own-data-682241488476)
2. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401.pdf)
3. [Retrieval Augmented Generation Using Llama2 And Falcon](https://medium.com/@scholarly360/retrieval-augmented-generation-using-llama2-and-falcon-ed26c7b14670)

## Contributors
- **Abhishek Singh** ([GitHub](https://github.com/Abhi2april), [LinkedIn](https://linkedin.com/in/abhishek-singh202220260204))
