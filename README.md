# Project-X

A Streamlit question-answering app for asking questions about Sahil's resume. The app combines a FAISS vector index, LangChain retrieval, and Azure/OpenAI chat models to answer resume-related questions through a simple web interface.

## What This Project Shows

- Building a resume Q&A assistant with retrieval augmented generation
- Using FAISS for local vector search
- Using Hugging Face embeddings for semantic retrieval
- Wrapping the workflow in a Streamlit multi-page app
- Storing resume data and generated vector indexes locally

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face embeddings
- Azure OpenAI / OpenAI-compatible chat models
- pandas

## Repository Structure

```text
.
├── Login.py                 # Streamlit entry page and lightweight user capture
├── pages/
│   └── QA Engine.py         # Main resume Q&A experience
├── components/
│   └── authentication.py
├── data/
│   ├── Sahil_Resume.pdf
│   ├── v1.csv
│   └── v2.xls
├── faiss_index_sap/         # Local FAISS index files
├── experiments/             # Earlier app and notebook experiments
├── requirements.txt
└── Dockerfile
```

## Run Locally

```bash
git clone https://github.com/sss2107/Project-X.git
cd Project-X
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run Login.py
```

## Configuration

The app expects model and API configuration through the Python config files and environment. For a safer setup, keep API keys outside source code:

```bash
export OPENAI_API_KEY=...
export AZURE_OPENAI_API_KEY=...
```

Then update the app code to read those values from `os.environ`.

## Security Follow-Up

This repository currently includes API-key style values in source files. Before deploying or sharing the app, rotate any exposed keys and replace hardcoded credentials with environment variables or a secret manager.

## Deployment Notes

The repository includes a `Dockerfile`, so the app can be containerized after secrets and environment configuration are cleaned up. For hosted Streamlit deployments, make sure the FAISS index files and resume data are available in the runtime environment.

## Improvement Ideas

- Remove hardcoded credentials
- Add a script for rebuilding the FAISS index from resume data
- Add clearer configuration docs for Azure OpenAI deployments
- Add sample questions and expected answer behavior
- Add basic tests for retrieval and response formatting
