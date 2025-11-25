
# DeepStack Character Info Extractor

## Overview
This project is a small RAG-style pipeline that builds a Chroma vector store from short story `.txt` files and exposes a CLI to extract structured JSON information about a character using Mistral’s API via LangChain.

**Pipeline Flow:**
1. Load raw stories  
2. Split into chunks  
3. Embed with `MistralAIEmbeddings`  
4. Store in Chroma  
5. Retrieve relevant chunks  
6. Call `ChatMistralAI` with a prompt that outputs a single JSON object.

---

## Setup

### 1. Clone and enter the project
```bash
git clone https://github.com/AdityaC784/DeepStack-Character-Info-Extractor.git
cd DeepStack-Character-Info-Extractor
```

### 2. Create and activate a virtual environment

**Windows (cmd):**
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux (bash / zsh):**
```bash
python -m venv .venv
source .venv/bin/activate
```
Using a dedicated virtual environment keeps this project’s dependencies isolated from your global Python installation.

### 3. Install dependencies
All dependencies are listed in `requirements.txt`.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

- `langchain`, `langchain-core`, `langchain-community`
- `langchain-mistralai`
- `langchain-chroma`, `langchain-text-splitters`
- `python-dotenv`

---

### 4. Configure Mistral API key

Both `embedding_pipe.py` and `retrieval_pipe.py` call `load_dotenv()`, so a `.env` file is expected at the project root.

1.Create and edit `.env`


2.Add your Mistral API key (replace ... with the real key):
```
MISTRAL_API_KEY="..."
```

---

### 5. Prepare story files

Place `.txt` stories in:

```
./stories
```

Example project layout:
```
project-root/
  cli.py
  embedding_pipe.py
  retrieval_pipe.py
  requirements.txt
  .env
  stories/
    a-mother.txt
    another-story.txt
  db/       # created automatically
```

Running `build_vector_store` reads `.txt` files, attaches metadata, splits them into chunks, and stores embeddings in Chroma.

---

## Usage

All interaction occurs through the Typer CLI in `cli.py`.

### 1. Compute embeddings (build or refresh DB)

```bash
python -m cli compute-embeddings --books-dir ./stories --db-path ./db

```

Arguments:

- `--books-dir` (`-b`): directory of `.txt` stories  
- `--db-path` (`-d`): directory where Chroma vector store is written

This:

- Ensures DB directory exists  
- Loads story files  
- Splits into chunks  
- Generates embeddings via `MistralAIEmbeddings`  
- Stores them in persistent Chroma

Run this anytime stories change.

---

### 2. Query for a character

Example:

```bash
python -m cli get-character-info "Devlin" --db-path ./db

```

This:

- Loads the Chroma store  
- Retrieves top-k chunks  
- Builds a second, more focused query  
- Ensures the character name exists  
- Calls `ChatMistralAI` with a custom prompt  
- Parses JSON output

Example output:

```json
{
  "name": "Devlin",
  "storyTitle": "A Mother",
  "summary": "...",
  "relations": [
    {"name": "Mr Kearney", "relation": "husband"}
  ],
  "characterType": "main"
}
```

To save the output to a file while testing:

```bash
python -m cli get-character-info "Devlin" --db-path ./db > devlin.json
```

---

## Notes & Troubleshooting

- **Must run embeddings first**  
  If `get-character-info` fails with missing vector store, run `compute-embeddings`.

- **Missing stories folder**  
  Ensure `./stories` exists and contains `.txt` files.

- **Mistral authentication errors**  
  Ensure:
  - `.env` exists  
  - `MISTRAL_API_KEY` is present  
  - Virtual environment is activated  

---

This README assumes basic knowledge of Python, the terminal, and virtual environments.


## Command EXECUTION(CLI):

### 1. Compute embeddings (build or refresh DB)

```bash
python -m cli compute-embeddings --books-dir ./stories --db-path ./db

```

### 2. Query for a character

Example:
Change the Character Name in place of Devlin

```bash
python -m cli get-character-info "Devlin" --db-path ./db

```

