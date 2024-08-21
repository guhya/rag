# eWIDE.AI

- Read PDF from directory
- Parse and split PDF into multiple chunks
- A chunk is tagged with {document_name}{page}{chunk_number}
- Load chunks to chroma DB after transformation with embedding function
- When user input is recieved, issue query to chroma DB to find all suitables chunk to send to LLM
- Use chunks as context, and together with input to from a prompt
- LLM should response with the context
