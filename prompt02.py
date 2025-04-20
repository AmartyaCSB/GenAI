import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()
llm = ChatGroq(model="llama-3-70b-versatile", temperature=0)

# Standard prompt for concise answers
standard_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question concisely: {question}."
)

# Chain-of-thought (CoT) prompt for step-by-step reasoning
cot_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question step by step concisely: {question}"
)

# Advanced Chain-of-Thought prompt
advanced_cot_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Solve the following problem step by step. For each step:
1. State what you're going to calculate
2. Write the formula you'll use (if applicable)
3. Perform the calculation
4. Explain the result

Question: {question}

Solution:
"""
)

# Creating chains
standard_chain = standard_prompt | llm
cot_chain = cot_prompt | llm
advanced_cot_chain = advanced_cot_prompt | llm

# Example question
question = "If a train travels 120 km in 2 hours, what is its average speed in km/h?"

# Invoke chains
standard_response = standard_chain.invoke(question).content
cot_response = cot_chain.invoke(question).content
advanced_cot_response = advanced_cot_chain.invoke(question).content

# Display responses
print("Standard Response:\n", standard_response)
print("\nChain-of-Thought Response:\n", cot_response)
print("\nAdvanced Chain-of-Thought Response:\n", advanced_cot_response)

logical_reasoning_prompt = PromptTemplate(
    input_variables=["scenario"],
    template="""
Analyze the following logical puzzle thoroughly. Follow these steps in your analysis:

List the Facts:

Summarize all the given information and statements clearly.
Identify all the characters or elements involved.

Identify Possible Roles or Conditions:

Determine all possible roles, behaviors, or states applicable to the characters or elements (e.g., truth-teller, liar).

Note the Constraints:

Outline any rules, constraints, or relationships specified in the puzzle.

Generate Possible Scenarios:

Systematically consider all possible combinations of roles or conditions for the characters or elements.
Ensure that all permutations are accounted for.

Test Each Scenario:

For each possible scenario:
Assume the roles or conditions you've assigned.
Analyze each statement based on these assumptions.
Check for consistency or contradictions within the scenario.

Eliminate Inconsistent Scenarios:

Discard any scenarios that lead to contradictions or violate the constraints.
Keep track of the reasoning for eliminating each scenario.

Conclude the Solution:

Identify the scenario(s) that remain consistent after testing.
Summarize the findings.

Provide a Clear Answer:

State definitively the role or condition of each character or element.
Explain why this is the only possible solution based on your analysis.

Scenario:

{scenario}

Analysis:
"""
)
