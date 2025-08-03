from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from database import query_journal_entries 
import re

# intialising llm
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

# defining prompt template
PROMPT_TEMPLATE = """
You are a reflective journaling assistant named MemoryJot. Your purpose is to help users understand their past thoughts and feelings based on their own journal entries, and to offer encouragement and support as a friend.

Use the following retrieved journal entries to answer the user's question **when relevant**. If the question is about a topic not covered in the journal entries, use your general knowledge and understanding to provide a helpful and friendly response.  You can also answer general knowledge questions (like "tell me a joke") even if the journal entries don't relate.

Retrieved Journal Entries:
{context}

User's Question:
{question}

Your Reflective Answer:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def is_general_question(query_text: str):
    """
    Determines if a query is a general question or related to journal entries.

    Args:
        query_text: The user's query.

    Returns:
        True if the query is likely a general question, False otherwise.
    """

    # List of keywords that indicate a general question
    general_keywords = ["what", "how", "tell", "joke", "weather", "news", "who", "why", "where", "suggest", "recommend", "advice"]

    # List of keywords that indicate a past experiences question
    past_experience_keywords = ["i", "my", "me", "mine", "last", "month", "week", "year", "ago", "felt", "did", "experience", "remember"]

    # Regular expression to match specific question formats
    regex_patterns = [
        r"^(tell me a joke)",
        r"^(what is the weather)",  
        r"^(what is the news)", 
        r"^(who is)",
        r"^(what are some)",
        r"^(why is)",
        r"^(where is)",  
        r"^(suggest me)",
        r"^(recommend me)", 
        r"^(can you give me advice)", 
    ]

    # Convert the query to lowercase for case-insensitive matching
    query_lower = query_text.lower()

    # if any general keywords are present in the query
    if any(keyword in query_lower for keyword in general_keywords):
        return True

    # if any regex patterns match the query
    if any(re.search(pattern, query_lower) for pattern in regex_patterns):
        return True

    # assume it's not a general question
    return False


def get_rag_response(query_text: str):
    """
    Generates a response using the RAG pattern.
    
    Args:
        query_text (str): The user's question.

    Returns:
        str: The LLM-generated response.
    """
    
    if is_general_question(query_text):
        #Create a simple prompt for answering general questions
        general_prompt = ChatPromptTemplate.from_template("Answer this general question in a friendly and helpful way: {question}")
        print("General Prompt:", general_prompt)
        print("Question (General):", query_text)
        response = llm.invoke(general_prompt.format(question=query_text))
        return response.content
    else:
        retrieved_docs = query_journal_entries(query_text, n_results=5)
        if not retrieved_docs:
            return "I couldn't find any relevant journal entries to answer your question."

        context_str = "\n\n---\n\n".join(retrieved_docs)
        print("Prompt Template:", prompt)
        print("Context:", context_str)
        print("Question (RAG):", query_text)
        final_prompt = prompt.format(context=context_str, question=query_text)
        response = llm.invoke(final_prompt)
        return response.content
