from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# Load Gemma 2 from Hugging Face
model_path = "C:/project/travel_ai/models/gemma-2b"  # you can try 9b model if GPU supports
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto"
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# 2. Wrap into LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

template = """
You are a friendly travel assistant. 
Use the following travel information to answer questions.
Travel Guide Data: {data}
User Question: {question}
Answer in a helpful, conversational way.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

while True:
    print("\n\n----------------------------")
    question = input("Ask your question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    result = chain.invoke({"data": [], "question": question})
    print(result)