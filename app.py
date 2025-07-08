import streamlit as st
from datetime import datetime
from database import add_journal_entry
from rag_core import get_rag_response

# page configuration
st.set_page_config(
    page_title="MemoryJot – Smart Reflective Journal",
    page_icon="📝",
    layout="wide"
)

# header
st.title("📝 MemoryJot – Your Smart Reflective Journal")
st.markdown("Capture your thoughts, recall your memories, and gain new insights.")

# main layout
col1,col2=st.columns(2)

# col1:add new entry
with col1:
    st.header("Add a New Journal Entry")
    new_entry_text=st.text_area("What's on your mind today?", height=200)
    if st.button("Save Entry"):
        if new_entry_text:
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata={"timestamp":timestamp}

            with st.spinner("Embedding and saving your entry..."):
                add_journal_entry(new_entry_text, metadata)
            
            st.success("Your entry has been saved successfully!")
        else:
            st.warning("Please write something before saving.")    



# col2: reflect on your past
with col2:
    st.header("Reflect on Your Past")
    query=st.text_input("Ask about your past entries...", placeholder="e.g., How did I feel about my project last month?")
    if st.button("Ask MemoryJot"):
        if query:
            with st.spinner("Searching your memories and generating a reflection..."):
                response=get_rag_response(query)
                st.info(response)
        else:
            st.warning("Please enter a question to reflect on.")        

# footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, LangChain, and OpenAI.")
