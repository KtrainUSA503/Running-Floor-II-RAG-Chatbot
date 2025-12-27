"""
KEITH Running Floor II Installation Assistant
RAG-powered chatbot for the Running Floor II Installation Manual

Built for Keith Manufacturing Company
"""

import streamlit as st
import openai
from pinecone import Pinecone
from typing import List, Tuple

# Page configuration
st.set_page_config(
    page_title="Running Floor II Installation Assistant",
    page_icon="🔧",
    layout="wide"
)

# Load Keith brand tokens
from pathlib import Path

def load_css(path: str) -> None:
    css = Path(path).read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css("brand/tokens.css")

# Optional: load Lato from Google Fonts (Streamlit doesn't ship it by default)
st.markdown("""
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Lato:wght@400;500;600;700&display=swap">
""", unsafe_allow_html=True)

# Custom CSS for Keith Manufacturing branding (token-based)
st.markdown("""
<style>
    /* Global typography */
    html, body, [class*="css"] {
        font-family: var(--keith-font);
        color: var(--keith-text);
    }

    /* App + page layout */
    .stApp {
        background-color: var(--keith-surface);
    }
    .main .block-container {
        max-width: 1120px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .main h1, .main h2, .main h3 {
        color: var(--keith-navy);
        letter-spacing: -0.01em;
    }

    /* Sidebar (override Streamlit theme so it matches) */
    [data-testid="stSidebar"] > div:first-child {
        background: var(--keith-bg);
        border-right: 1px solid var(--keith-border);
    }
    [data-testid="stSidebar"] h3 {
        color: var(--keith-navy);
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span {
        color: var(--keith-text);
    }

    /* Header */
    .main-header {
        background: linear-gradient(90deg, var(--keith-navy) 0%, var(--keith-blue) 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: var(--keith-shadow);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin: 6px 0 0 0;
    }

    /* Links */
    a, a:visited {
        color: var(--keith-navy);
    }
    a:hover {
        color: var(--keith-blue);
    }

    /* Sources */
    .source-box {
        background-color: var(--keith-surface-2);
        padding: 10px;
        border-radius: 6px;
        border-left: 4px solid var(--keith-navy);
        margin: 5px 0;
        font-size: 0.9em;
    }

    /* Chat message cards */
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .user-message {
        background-color: var(--keith-surface-2);
        border: 1px solid var(--keith-border);
    }
    .assistant-message {
        background-color: white;
        border: 1px solid var(--keith-border);
    }

    /* Buttons (Streamlit) */
    .stButton > button {
        border-radius: var(--keith-radius-pill);
        padding: 10px 18px;
        font-weight: 600;
        border: 1px solid var(--keith-border);
        transition: background-color 120ms ease, border-color 120ms ease, transform 80ms ease;
    }
    .stButton > button:active {
        transform: translateY(1px);
    }
    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background: var(--keith-navy);
        color: white;
        border-color: var(--keith-navy);
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--keith-blue);
        border-color: var(--keith-blue);
    }
    .stButton > button[kind="primary"]:focus {
        box-shadow: 0 0 0 4px var(--keith-focus);
    }
    /* Secondary buttons (use for example question “chips”) */
    .stButton > button[kind="secondary"] {
        background: var(--keith-bg);
        color: var(--keith-navy);
        border-color: var(--keith-border);
    }
    .stButton > button[kind="secondary"]:hover {
        background: var(--keith-surface-2);
        border-color: var(--keith-blue);
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        border-radius: 14px;
        border: 1px solid var(--keith-border);
    }
    [data-testid="stChatInput"] textarea:focus {
        box-shadow: 0 0 0 4px var(--keith-focus);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pinecone_index" not in st.session_state:
    st.session_state.pinecone_index = None

def init_pinecone():
    """Initialize Pinecone connection."""
    if st.session_state.pinecone_index is None:
        try:
            pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
            st.session_state.pinecone_index = pc.Index(st.secrets.get("PINECONE_INDEX", "running-floor-manual"))
        except Exception as e:
            st.error(f"Failed to initialize Pinecone: {e}")
            return None
    
    return st.session_state.pinecone_index

def get_embedding(text: str) -> List[float]:
    """Get embedding for a text using OpenAI."""
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        dimensions=1536
    )
    return response.data[0].embedding

def search_knowledge_base(query: str, index, top_k: int = 5) -> List[dict]:
    """Search Pinecone for relevant documents."""
    query_embedding = get_embedding(query)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return results.matches

def build_context(matches: List[dict]) -> Tuple[str, List[dict]]:
    """Build context string from search results."""
    context_parts = []
    sources = []
    
    for match in matches:
        if match.score > 0.7:  # Only include relevant matches
            text = match.metadata.get("text", "")
            page = match.metadata.get("page", "N/A")
            context_parts.append(text)
            sources.append({
                "text": text[:200] + "..." if len(text) > 200 else text,
                "page": page + 1,  # Convert to 1-indexed
                "score": round(match.score, 3)
            })
    
    return "\n\n".join(context_parts), sources

def get_chat_response(query: str, context: str, chat_history: List[dict]) -> str:
    """Get response from OpenAI using RAG context."""
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    system_prompt = """You are the KEITH Running Floor II Installation Assistant, an expert AI assistant 
specializing in the installation and maintenance of KEITH Walking Floor® unloading systems.

Your role is to help installers, technicians, and operators with questions about:
- Installation procedures and best practices
- Troubleshooting common issues
- Component specifications and requirements
- Safety guidelines and warnings
- Maintenance recommendations

Guidelines:
1. Always base your answers on the provided context from the installation manual
2. If the context doesn't contain enough information, say so clearly
3. Highlight important safety warnings when relevant
4. Use clear, technical language appropriate for skilled installers
5. Reference specific page numbers or sections when helpful
6. If asked about something outside the manual's scope, acknowledge the limitation

Remember: Installing the WALKING FLOOR® system requires alterations to trailers. 
Always emphasize safety and proper procedures."""

    # Build messages with history
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add relevant chat history (last 6 exchanges)
    for msg in chat_history[-12:]:
        messages.append(msg)
    
    # Add the current query with context
    user_message = f"""Based on the following context from the Running Floor II Installation Manual:

---
{context}
---

User Question: {query}

Please provide a helpful, accurate response based on the manual content."""

    messages.append({"role": "user", "content": user_message})
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔧 Running Floor II Installation Assistant</h1>
        <p>AI-powered support for KEITH Walking Floor® system installation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://www.keithwalkingfloor.com/wp-content/themes/theme/images/logo.png", width=200)
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This assistant helps with questions about installing 
        the KEITH Running Floor II® unloading system.
        
        **Topics covered:**
        - Trailer preparations
        - Drive unit installation
        - Flooring & seals
        - Hydraulic systems
        - Troubleshooting
        """)
        
        st.markdown("---")
        st.markdown("### Quick References")
        st.markdown("""
        - Installation time: 35-100 hours
        - System type: 8" stroke, 3½" flooring
        - Contact: 800-547-6161
        """)
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize Pinecone
    index = init_pinecone()
    
    if index is None:
        st.error("⚠️ Unable to connect to the knowledge base. Please check your configuration.")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for source in message["sources"]:
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Page {source['page']}</strong> (Relevance: {source['score']})
                            <br><em>{source['text']}</em>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the Running Floor II installation..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching manual and generating response..."):
                # Search knowledge base
                matches = search_knowledge_base(prompt, index)
                context, sources = build_context(matches)
                
                if not context:
                    response = """I couldn't find specific information about that in the Running Floor II 
                    Installation Manual. Could you rephrase your question, or ask about:
                    - Trailer preparation and alignment
                    - Drive unit installation (center frame or frameless)
                    - Sub-deck and flooring installation
                    - Hydraulic tubing setup
                    - Seal installation procedures"""
                    sources = []
                else:
                    # Get chat response
                    chat_history = [
                        {"role": m["role"], "content": m["content"]} 
                        for m in st.session_state.messages[:-1]
                    ]
                    response = get_chat_response(prompt, context, chat_history)
                
                st.markdown(response)
                
                if sources:
                    with st.expander("📚 View Sources"):
                        for source in sources:
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Page {source['page']}</strong> (Relevance: {source['score']})
                                <br><em>{source['text']}</em>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Save assistant message with sources
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sources": sources
        })

# Example questions section
def show_example_questions():
    st.markdown("### 💡 Try these questions:")
    
    examples = [
        "How do I align the drive unit in a center frame trailer?",
        "What are the steps for installing floor seals?",
        "How should I route hydraulic tubing?",
        "What's the recommended torque for floor bolts?",
        "How do I prepare the trailer before installation?",
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"❓ {example}", key=f"example_{i}", type="secondary"):
                st.session_state.example_question = example

if __name__ == "__main__":
    main()
    
    # Show example questions if chat is empty
    if not st.session_state.messages:
        show_example_questions()