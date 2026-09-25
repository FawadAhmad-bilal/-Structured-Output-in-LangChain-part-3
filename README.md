# Prompt Engineering — LangChain Structured Output

Code and notes from the **CampusX** prompt engineering / LangChain series, covering how to get reliable, schema-validated output from LLMs instead of raw free-text responses.

## What's in here

This repo works through three approaches to structured output in LangChain, using a Hugging Face-hosted `meta-llama/Llama-3.1-8B-Instruct` model via `HuggingFaceEndpoint` + `ChatHuggingFace`:

1. **TypedDict-based structured output** — using `with_structured_output()` with a `TypedDict` schema.
2. **Pydantic-based structured output** — attempting `with_structured_output()` with a Pydantic `BaseModel` (and why this fails for some Hugging Face-hosted models that don't support native function/tool calling).
3. **Manual parsing with `PydanticOutputParser`** — the reliable fallback: prompting the model directly for JSON matching a Pydantic schema, then parsing and validating the raw text output. This is the approach that worked consistently in testing.

## Key concepts covered

- Difference between `TypedDict` and Pydantic `BaseModel` for schema definition
- Why `with_structured_output()` depends on the model provider supporting function/tool calling
- Using `Literal` for fixed-choice fields (e.g. sentiment: positive/negative/neutral) vs `list[str]` for fields that can hold multiple independent values (e.g. pros, cons, key themes)
- `Optional[...]` fields for values that may not be present in every input
- `PydanticOutputParser.get_format_instructions()` to auto-generate schema instructions for the prompt
- Chaining with LangChain's pipe syntax: `prompt | model | parser`
- Tuning `max_new_tokens` to avoid truncated/invalid JSON output

## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1   # Windows PowerShell
pip install langchain langchain-huggingface langchain-core python-dotenv pydantic
```

Create a `.env` file in the project root with your Hugging Face token:

```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

**Never commit `.env` or paste tokens into chat/commits — add `.env` to `.gitignore`.**

## Example use case

The example task used throughout: extracting structured review data (key themes, summary, sentiment, pros, cons, reviewer name) from a free-text product review, validated against a Pydantic schema.

## Notes

- `meta-llama/Llama-3.1-8B-Instruct` is a gated model on Hugging Face — the license must be accepted on your HF account before the endpoint will work.
- If `with_structured_output()` raises `NotImplementedError`, it means the chosen method (e.g. function calling) isn't supported for this model/integration — fall back to the manual `PydanticOutputParser` chain instead.

## Status

working
