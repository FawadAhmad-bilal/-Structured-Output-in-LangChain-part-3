from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import TypeAdapter

load_dotenv()


class Review(TypedDict):
    summary: Annotated[str, "A concise summary of the whole review"]
    sentiment: Annotated[
        Literal["positive", "negative", "mixed"],
        "Overall sentiment of the review",
    ]
    pros: Annotated[list[str], "Positive points the reviewer mentioned"]
    cons: Annotated[list[str], "Negative points the reviewer mentioned"]


model = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
)
llm = ChatHuggingFace(llm=model)

review_text = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by fawad ahmad bilal
"""

instructions = """Return ONLY a JSON object with exactly these keys:
{
  "summary": "a concise summary of the whole review",
  "sentiment": "positive" or "negative" or "mixed",
  "pros": ["positive point", "another positive point"],
  "cons": ["negative point", "another negative point"]
}
Do not write any text before or after the JSON."""

prompt = f"Analyze the following product review.\n\n{instructions}\n\nReview:\n{review_text}"

# Step A: raw reply from the model
raw = llm.invoke(prompt)
print("RAW REPLY:\n", raw.content, "\n")

# Step B: text -> dictionary -> check it against your TypedDict
try:
    data = JsonOutputParser().parse(raw.content)
    result = TypeAdapter(Review).validate_python(data)

    print(result["summary"])
    print(result["sentiment"])
    print(result["pros"])
    print(result["cons"])
except Exception as e:
    print("Parsing or validation failed:", e)