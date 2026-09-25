from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import JsonOutputParser,PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Annotated,Literal,Optional
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()

class review(BaseModel):

    key_theme:list[str]=Field(description='write down all the key themes discuss in the reviewer list')
    summary:list[str]=Field(description='a brief summary on the review')
    sentiment:Literal['positive', 'negative', 'neutral']=Field(description='return the sentiment review either negative,positive,neutral')
    pros: Optional[list[str]]=Field(default=None,description='write down all the prons inside the list')
    cons: Optional[list[str]]=Field(default=None,description='write down all the cons inside the list')
    name:Optional[list[str]]=Field(default=None,description='write down the reviewr name')


llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation',
    temperature=0.8,
    max_new_tokens=512
)
model=ChatHuggingFace(llm=llm)

paras=PydanticOutputParser(pydantic_object=review)
# structured_model=model.with_structured_output(review)

prompt=PromptTemplate(
     template="Analyze the following review and respond ONLY with valid JSON, no extra text.\n{format_instructions}\n\nReview:\n{review_text}",
    input_variables=["review_text"],
    partial_variables={"format_instructions": paras.get_format_instructions()}
)

chain=prompt|model|paras

result=chain.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh""")

print(result)
print(result.summary)
print(result.sentiment)