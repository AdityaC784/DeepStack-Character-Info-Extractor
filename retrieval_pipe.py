from collections import Counter
from pathlib import Path
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = """
You are an information extraction assistant.
    Given story excerpts and a character name, extract a JSON object with:
    - name
    - storyTitle
    - summary
    - relations: list of objects with keys "name" and "relation"
    - characterType

    The character name in the query may be a first name, last name, maiden name,
    married name, title + surname (e.g. "Mr Jack"), or a nickname.
    If the story clearly indicates that two names refer to the same person,
    treat them as the same character.

    Use all relevant mentions (including variants of the name) to build the summary
    and relations for that one character.

    Always return a single JSON object and nothing else.
"""


USER_PROMPT = """
Character name: {character_name}

Story excerpts:
{context}

Return ONLY the JSON object, no explanations.
"""


def _load_vector_store(db_path: str) -> Chroma:
    db_dir = Path(db_path)

    if not db_dir.exists():
        raise FileNotFoundError(
            f"Vector store directory {db_dir} does not exist. "
            "Run the embedding_pipe.py first."
        )
    else:
        print(f"[DEBUG] Loading vector store from: {db_dir.resolve()}")

    embeddings = MistralAIEmbeddings()
    vectordb = Chroma(
        persist_directory=str(db_dir),
        embedding_function=embeddings,
    )
    return vectordb


def _guess_story_title(docs) -> str | None:

    """Pick the most frequent story_title from retrieved docs."""

    titles = [d.metadata.get("story_title") for d in docs if d.metadata.get("story_title")]

    if not titles:
        return None
    
    counts = Counter(titles)
    best_story_title, _ = counts.most_common(1)[0]
    print(f"[DEBUG] Best matching story_title: {best_story_title} [from first-pass retrieval] ")
    return best_story_title


def get_character_info(character_name: str, db_path: str) -> str:

    character_name = character_name.strip()
    
    if not character_name:
        return '{"error": "Character name is empty."}'

    vectordb = _load_vector_store(db_path)

    print(f"[DEBUG] First-pass retrieval for character: {character_name!r}")
    first_retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    first_docs = first_retriever.invoke(character_name)

    if not first_docs:
        print("[DEBUG] No documents returned in first-pass retrieval.")
        return '{"error": "Character not found in any story."}'

    best_story_title = _guess_story_title(first_docs)


    if not best_story_title:
        print("[DEBUG] Could not determine best story_title, using first-pass docs only.")
        story_docs = first_docs
    else:
       
        print(f"[DEBUG] Second-pass retrieval filtered to story_title={best_story_title!r}")

        filtered_retriever = vectordb.as_retriever(
            search_kwargs={
                "k": 300,  # high enough to cover all chunks for that story
                "filter": {"story_title": best_story_title},
            }
        )
        story_docs = filtered_retriever.invoke(character_name)

        if not story_docs:
            print("[DEBUG] No documents found in second-pass filtered retrieval; falling back to first-pass docs.")
            story_docs = first_docs

    text_blob = "\n\n".join(d.page_content for d in story_docs)
    if character_name.lower() not in text_blob.lower():
        return '{"error": "Character not found"}'
    
    context = "\n\n---\n\n".join(
        f"Title: {d.metadata.get('story_title')}\n{d.page_content}"
        for d in story_docs
    )

 
    llm = ChatMistralAI(model="mistral-small-2506",temperature=0.1)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)]
    )
    chain = prompt | llm

    print("[DEBUG] Sending context to LLM for structured extraction...")
    resp = chain.invoke({"character_name": character_name, "context": context})
    return resp.content
