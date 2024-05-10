# Advanced RAG using LLMSmith and FastAPI

## Setup

- Create a virtual environment
- Run `pip install -r requirements.txt` or `poetry install` (if using `poetry`)
- Run Qdrant vector database locally
- Create a `.env` file and copy the contents of `.env.template` into `.env`
- Update `.env` with the required values
- OPTIONAL: Set `LLMSMITH__DEBUG=true` in `.env` to see debug logs for `LLMSmith`

## Insert sample data into Qdrant

- Run `insert-sample-data.py` to insert embeddings of a small dataset related to Harvard University into Qdrant DB.

The `stanford-dataset-dev-v2.0.json` inside `test-data` folder contains data related to many other topics.
If you need to embed dataset related to some other topic, just change `title_to_insert` variable in `insert-sample-data.py` accordingly.

## Start server and try the /chat API

Run `make run-dev` to start the FastAPI app
Use the below curl command to talk with the RAG chatbot.

`curl -X POST --header "Content-Type: application/json" -d '{"content": "in 1846 who's natural history lectures were acclaimed in New York and Harvard?"}' localhost:8000/api/chat`

This will give the response in the below format

`{"content": "In 1846, the natural history lectures acclaimed in New York and Harvard were given by Louis Agassiz."}`
