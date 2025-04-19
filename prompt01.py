# sentiment_classification.py

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(model="llama-3-8b-versatile", temperature=0)

def few_shot_sentiment_classification(input_text):
    few_shot_prompt = PromptTemplate(
        input_variables=["input_text"],
        template="""
Classify the sentiment as Positive, Negative, or Neutral.

Examples:
Text: I love this product! It's amazing.
Sentiment: Positive

Text: This movie was terrible. I hated it.
Sentiment: Negative

Text: The weather today is okay.
Sentiment: Neutral

Now, classify the following:
Text: {input_text}
Sentiment:
"""
    )

    chain = few_shot_prompt | llm
    result = chain.invoke(input_text).content

    result = result.strip()
    if ':' in result:
        result = result.split(':')[1].strip()

    return result



def multi_task_few_shot(input_text, task):
    few_shot_prompt = PromptTemplate(
        input_variables=["input_text", "task"],
        template="""
Perform the specified task on the given text.

Examples:
Text: I love this product! It's amazing.
Task: sentiment
Result: Positive

Text: Bonjour, comment allez-vous?
Task: language
Result: French

Now, perform the following task:
Text: {input_text}
Task: {task}
Result:
"""
    )

    chain = few_shot_prompt | llm
    return chain.invoke({"input_text": input_text, "task": task}).content

def in_context_learning(task_description, examples, input_text):
    example_text = "\n".join([f"Input: {e['input']}\nOutput: {e['output']}\n" for e in examples])

    in_context_prompt = PromptTemplate(
        input_variables=["task_description", "examples", "input_text"],
        template="""
Task: {task_description}

Examples:
{examples}

Now, perform the task on the following input:
Input: {input_text}
Output:
"""
    )

    chain = in_context_prompt | llm
    return chain.invoke({
        "task_description": task_description,
        "examples": example_text,
        "input_text": input_text
    }).content
