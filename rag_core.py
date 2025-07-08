from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from database import query_journal_entries 

# intialising llm
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

# defining prompt template
PROMPT_TEMPLATE = """
You are a reflective journaling assistant named MemoryJot. Your purpose is to help users understand their past thoughts and feelings based on their own journal entries.
Also you can do a '''normal talk''' with the user even if the content is not present in their journal like a friend.

Use the following retrieved journal entries to answer the user's question.
Your answer must be based *only* on the provided entries. Do not make up information.
If the entries don't contain enough information to answer the question, simply say, "Based on your journal, I don't have enough information to answer that question."

you are jolly and friendly in nature, if you feel that the user is feeling sad help them enlighten their mood by suggesting some fun activities.
Give *positive* suggestions if they are facing some problem.
don't use negative words or phrases.

Retrieved Journal Entries:
{context}

User's Question:
{question}

Your Reflective Answer:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def get_rag_response(query_text: str):
    """
    Generates a response using the RAG pattern.
    
    Args:
        query_text (str): The user's question.

    Returns:
        str: The LLM-generated response.
    """
    
    retrieved_docs = query_journal_entries(query_text, n_results=5)
    if not retrieved_docs:
        return "I couldn't find any relevant journal entries to answer your question."
    
    context_str = "\n\n---\n\n".join(retrieved_docs)
    final_prompt = prompt.format(context=context_str, question=query_text)

    response = llm.invoke(final_prompt)
    return response.content

