run-dev:
	python -m uvicorn --reload rag_llmsmith_fastapi.main:app --port 8000