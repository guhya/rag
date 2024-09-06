import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma_mysql"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
Explain why you think your answer is related to the question.

"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--llm", action='store_true', help="Send prompt to LLM.")
    
    args = parser.parse_args()
    query_text = args.query_text

    query_rag(query_text, args.llm)


def query_rag(query_text: str, send_to_llm: bool):
    # Prepare the DB.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=10)
    doc_sets = set()
    for doc, _score in results:
        doc_sets.add(doc.metadata.get("mysql_id", None))


    # Send to LLM if flag is set
    if send_to_llm:
        # Define the metadata criteria
        metadata_criteria = {"mysql_id": { "$in" : list(doc_sets)}}
        retriever = db.get(where=metadata_criteria)
        docs = retriever.get("documents")
        metadatas = retriever.get("metadatas")

        i = 0
        combined_docs = [];
        for doc in metadatas:
            tmp = metadatas[i]
            tmp["content"] = docs[i]
            i += 1
            combined_docs.append(tmp)
            
        combined_docs_sorted = sorted(combined_docs, key=lambda x: (x['mysql_id'], x['chunk_index']))

        mysql_id = -1
        context_text = ""
        for doc in combined_docs_sorted:
            if mysql_id != doc["mysql_id"]:
                context_text += "\n\n---\n\n"
            context_text += doc["content"] + " "
            mysql_id = doc["mysql_id"]

        print(f"Sending prompt to LLM..")

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        print(prompt)

        model = Ollama(model="llama3.1:8b-instruct-q8_0")
        response_text = model.invoke(prompt)

    formatted_response = f"Response: {response_text}\n\nSources: {doc_sets}"
    print(formatted_response)

    return formatted_response


if __name__ == "__main__":
    main()
