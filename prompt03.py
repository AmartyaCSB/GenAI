import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import random
from collections import Counter

load_dotenv()
llm = ChatGroq(model="llama-3-70b-versatile", temperature=0)

def generate_multiple_paths(problem, num_paths=3):
    """
    Generate multiple reasoning paths for a given problem.

    Args:
        problem (str): The problem statement.
        num_paths (int): Number of reasoning paths to generate.

    Returns:
        list: A list of generated reasoning paths.
    """

    prompt_template = PromptTemplate(
        input_variables=["problem", "path_number"],
        template="""Solve the following problem using a unique approach. This is reasoning path {path_number}.
Problem: {problem}
Reasoning path {path_number}:"""
    )

    paths = []
    for i in range(num_paths):
        chain = prompt_template | llm
        response = chain.invoke({"problem": problem, "path_number": i+1}).content
        paths.append(response)

    return paths

def select_final_answer(paths):
    """
    Select the final answer based on majority vote from multiple reasoning paths.

    Args:
        paths (list): List of reasoning paths.

    Returns:
        str: Final answer selected by majority vote.
    """
    answers = [path.strip().split('\n')[-1] for path in paths]
    answer_counts = Counter(answers)
    most_common_answer, _ = answer_counts.most_common(1)[0]

    return most_common_answer

# Example usage
problem = "If a train travels 120 km in 2 hours, what is its average speed in km/h?"

# Generate multiple reasoning paths
reasoning_paths = generate_multiple_paths(problem, num_paths=5)

# Select final answer based on majority voting
final_answer = select_final_answer(reasoning_paths)

# Display each reasoning path
for i, path in enumerate(reasoning_paths, 1):
    print(f"Reasoning Path {i}:\n{path}\n")

# Display final selected answer
print(f"Final Answer:\n{final_answer}")


# # Self consistency implementation using multiple reasoning paths

# import os
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import random
# from collections import Counter

# load_dotenv()
# llm = ChatGroq(model="llama-3-70b-versatile", temperature=0)

# def generate_multiple_paths(problem, num_paths=3):
#     """
#     Generate multiple reasoning paths for a given problem.

#     Args:
#         problem (str): The problem statement.
#         num_paths (int): Number of reasoning paths to generate.

#     Returns:
#         list: A list of generated reasoning paths.
#     """

#     prompt_template = PromptTemplate(
#         input_variables=["problem", "path_number"],
#         template="""Solve the following problem using a unique approach. This is reasoning path {path_number}.
# Problem: {problem}
# Reasoning path {path_number}:"""
#     )

#     paths = []
#     for i in range(num_paths):
#         chain = prompt_template | llm
#         response = chain.invoke({"problem": problem, "path_number": i+1}).content
#         paths.append(response)

#     return paths

# # Example problem:
# problem = "A ball is thrown upwards with an initial velocity of 20 m/s. How high will it go?"
# paths = generate_multiple_paths(problem)

# # Displaying generated reasoning paths
# for idx, path in enumerate(paths, 1):
#     print(f"Reasoning Path {idx}:\n{path}\n")
