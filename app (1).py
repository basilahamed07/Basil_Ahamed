# """
# Git Repository Q&A - Streamlit Application
# A POC for intelligent code repository question answering
# """
"""
Git Repository Q&A - Streamlit Application
A POC for intelligent code repository question answering
"""
import streamlit as st
import logging
import os
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Git Repository Q&A",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import components
from config import config
from git_analyzer import GitAnalyzer, RepositoryProcessor
from embeddings import EmbeddingManager
from vector_store import VectorStoreManager
from query_processor import CodeQAProcessor
from hierarchical_query_engine import ConversationalHierarchicalEngine
from file_summarizer import FileSummaryGenerator


def init_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.repos_processed = []
        st.session_state.conversation_history = []
        st.session_state.query_engine = None
        st.session_state.processing = False
        st.session_state.current_repo = None


def get_components():
    """Initialize and cache components"""
    if 'components' not in st.session_state:
        # Initialize components
        git_analyzer = GitAnalyzer(
            allowed_exts=config.allowed_extensions,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap
        )
        
        embedding_manager = EmbeddingManager(config)
        
        vector_store = VectorStoreManager(
            persist_directory=config.chroma.persist_directory,
            collection_name=config.chroma.collection_name
        )
        
        llm_processor = CodeQAProcessor(
            azure_endpoint=config.azure_openai.endpoint,
            azure_api_key=config.azure_openai.api_key,
            deployment_name=config.azure_openai.chat_deployment,
            api_version=config.azure_openai.api_version
        )
        
        # Initialize file summarizer
        file_summarizer = FileSummaryGenerator(
            azure_endpoint=config.azure_openai.endpoint,
            azure_api_key=config.azure_openai.api_key,
            deployment_name=config.azure_openai.chat_deployment,
            api_version=config.azure_openai.api_version
        )
        
        repo_processor = RepositoryProcessor(
            git_analyzer=git_analyzer,
            embedding_manager=embedding_manager,
            vector_store_manager=vector_store,
            file_summarizer=file_summarizer
        )
        
        query_engine = ConversationalHierarchicalEngine(
            embedding_manager=embedding_manager,
            vector_store_manager=vector_store,
            llm_processor=llm_processor
        )
        
        st.session_state.components = {
            'git_analyzer': git_analyzer,
            'embedding_manager': embedding_manager,
            'vector_store': vector_store,
            'llm_processor': llm_processor,
            'file_summarizer': file_summarizer,
            'repo_processor': repo_processor,
            'query_engine': query_engine
        }
        
        st.session_state.query_engine = query_engine
        st.session_state.initialized = True
    
    return st.session_state.components


def render_sidebar():
    """Render sidebar with configuration options"""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Azure OpenAI Status
        st.subheader("Azure OpenAI")
        if config.validate_azure_config():
            st.success("✅ Configured")
        else:
            st.error("❌ Missing configuration")
            st.info("Please set Azure OpenAI environment variables")
        
        st.divider()
        
        # Repository Management
        st.subheader("📁 Repositories")
        
        components = get_components()
        vector_store = components['vector_store']
        
        # Get stored repos
        repos = vector_store.store.get_all_repos()
        
        if repos:
            st.write(f"**Indexed Repositories:** {len(repos)}")
            for repo in repos:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"• {repo}")
                with col2:
                    if st.button("🗑️", key=f"del_{repo}", help=f"Delete {repo}"):
                        if vector_store.clear_repository(repo):
                            st.success(f"Deleted {repo}")
                            st.rerun()
        else:
            st.info("No repositories indexed yet")
        
        st.divider()
        
        # Collection Stats
        st.subheader("📊 Statistics")
        stats = vector_store.store.get_collection_stats()
        st.metric("Total Chunks", stats.get('total_chunks', 0))
        
        if stats.get('sample_languages'):
            st.write("**Languages:**")
            st.text(", ".join(stats['sample_languages'][:5]))
        
        st.divider()
        
        # Settings
        st.subheader("🔧 Settings")
        
        embedding_model = st.radio(
            "Embedding Model",
            options=["Azure OpenAI", "CodeBERT"],
            index=0,
            help="Choose embedding model: Azure OpenAI (faster) or CodeBERT (better for code)"
        )
        st.session_state.use_codebert = (embedding_model == "CodeBERT")
        
        top_k = st.slider(
            "Context chunks",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant code chunks to retrieve"
        )
        st.session_state.top_k = top_k
        
        st.divider()
        
        # Reset Options
        if st.button("🔄 Reset All Data", type="secondary"):
            if st.session_state.get('confirm_reset', False):
                vector_store.reset()
                st.session_state.conversation_history = []
                st.session_state.confirm_reset = False
                st.success("All data reset!")
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm reset")


def render_repo_input():
    """Render repository input section"""
    st.header("📥 Index Repository")
    
    with st.expander("Add New Repository", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            repo_url = st.text_input(
                "Repository URL",
                placeholder="https://github.com/owner/repo",
                help="GitHub, Azure DevOps, or GitLab repository URL"
            )
        
        with col2:
            branch = st.text_input(
                "Branch",
                value="main",
                help="Branch to clone"
            )
        
        col3, col4 = st.columns([2, 1])
        
        with col3:
            token = st.text_input(
                "Access Token (optional)",
                type="password",
                help="Personal access token for private repositories"
            )
        
        with col4:
            embedding_model_index = st.radio(
                "Embedding for indexing",
                options=["Azure OpenAI", "CodeBERT"],
                index=0,
                help="Choose embedding model for indexing"
            )
            use_codebert_index = (embedding_model_index == "CodeBERT")
        
        if st.button("🚀 Index Repository", type="primary", disabled=st.session_state.processing):
            if repo_url:
                process_repository(repo_url, token, branch, use_codebert_index)
            else:
                st.error("Please enter a repository URL")


def process_repository(repo_url: str, token: str, branch: str, use_codebert: bool):
    """Process and index a repository"""
    st.session_state.processing = True
    components = get_components()
    repo_processor = components['repo_processor']
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(stage: str, progress: float, message: str):
        progress_bar.progress(progress)
        status_text.text(f"[{stage.upper()}] {message}")
    
    try:
        result = repo_processor.process_repository(
            repo_url=repo_url,
            token=token if token else None,
            branch=branch,
            use_codebert=use_codebert,
            progress_callback=update_progress
        )
        
        if result['status'] == 'success':
            st.success(f"✅ Successfully indexed: {result['repo_name']}")
            
            # Show statistics
            chunk_stats = result.get('chunk_stats', {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Files Processed", chunk_stats.get('files_processed', 0))
            with col2:
                st.metric("Chunks Created", chunk_stats.get('chunks_created', 0))
            with col3:
                languages = chunk_stats.get('languages', [])
                st.metric("Languages", len(languages))
            
            if languages:
                st.info(f"Languages detected: {', '.join(languages[:10])}")
            
            st.session_state.repos_processed.append(result['repo_name'])
            st.session_state.current_repo = result['repo_name']
        else:
            st.error(f"❌ Error: {result.get('message', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"❌ Error processing repository: {str(e)}")
        logger.exception("Repository processing error")
    
    finally:
        st.session_state.processing = False
        progress_bar.empty()
        status_text.empty()


# def render_qa_interface():
#     """Render the Q&A interface"""
#     st.header("💬 Ask Questions About Your Code")
    
#     components = get_components()
#     vector_store = components['vector_store']
    
#     # Check if any repos are indexed
#     repos = vector_store.store.get_all_repos()
    
#     if not repos:
#         st.info("👆 Please index a repository first to start asking questions")
#         return
    
#     # Repository filter
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         selected_repo = st.selectbox(
#             "Filter by Repository",
#             options=["All Repositories"] + repos,
#             index=0
#         )
#     with col2:
#         if st.button("🔄 Clear History"):
#             st.session_state.conversation_history = []
#             st.session_state.query_engine.clear_history()
#             st.rerun()
    
#     repo_filter = None if selected_repo == "All Repositories" else selected_repo
    
#     # Chat interface
#     st.divider()
    
#     # Display conversation history
#     for i, exchange in enumerate(st.session_state.conversation_history):
#         with st.chat_message("user"):
#             st.write(exchange['question'])
        
#         with st.chat_message("assistant"):
#             st.write(exchange['answer'])
            
#             # Show sources
#             if exchange.get('sources'):
#                 with st.expander("📚 View Sources"):
#                     for source in exchange['sources']:
#                         st.code(f"File: {source['file_path']}\n"
#                                f"Type: {source['chunk_type']}\n"
#                                f"Lines: {source['start_line']}-{source['end_line']}\n"
#                                f"Score: {source['score']:.3f}")
    
#     # Query input
#     query = st.chat_input("Ask a question about the code...")
    
#     if query:
#         # Add user message
#         with st.chat_message("user"):
#             st.write(query)
        
#         # Process query
#         with st.chat_message("assistant"):
#             with st.spinner("Searching and analyzing code..."):
#                 try:
#                     result = st.session_state.query_engine.query(
#                         question=query,
#                         top_k=st.session_state.get('top_k', 5),
#                         repo_filter=repo_filter,
#                         use_codebert=st.session_state.get('use_codebert', False)
#                     )
                    
#                     st.write(result.answer)
                    
#                     # Show confidence and sources
#                     st.caption(f"Confidence: {result.confidence:.2%}")
                    
#                     if result.sources:
#                         with st.expander("📚 View Sources"):
#                             for source in result.sources:
#                                 meta = source['metadata']
#                                 st.code(f"File: {meta.get('file_path', 'unknown')}\n"
#                                        f"Type: {meta.get('chunk_type', 'code')}\n"
#                                        f"Name: {meta.get('name', 'N/A')}\n"
#                                        f"Lines: {meta.get('start_line', '?')}-{meta.get('end_line', '?')}\n"
#                                        f"Score: {source['score']:.3f}")
                    
#                     # Store in history
#                     st.session_state.conversation_history.append({
#                         'question': query,
#                         'answer': result.answer,
#                         'sources': [
#                             {
#                                 'file_path': s['metadata'].get('file_path', 'unknown'),
#                                 'chunk_type': s['metadata'].get('chunk_type', 'code'),
#                                 'start_line': s['metadata'].get('start_line', 0),
#                                 'end_line': s['metadata'].get('end_line', 0),
#                                 'score': s['score']
#                             }
#                             for s in result.sources
#                         ],
#                         'confidence': result.confidence
#                     })
                    
#                 except Exception as e:
#                     st.error(f"Error processing query: {str(e)}")
#                     logger.exception("Query processing error")
def render_qa_interface():
    """Render the Q&A interface"""
    st.header("💬 Ask Questions About Your Code")
    
    components = get_components()
    vector_store = components['vector_store']
    
    # Check if any repos are indexed
    repos = vector_store.store.get_all_repos()
    
    if not repos:
        st.info("👆 Please index a repository first to start asking questions")
        return
    
    # Repository filter
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_repo = st.selectbox(
            "Filter by Repository",
            options=["All Repositories"] + repos,
            index=0
        )
    with col2:
        if st.button("🔄 Clear History"):
            st.session_state.conversation_history = []
            st.session_state.query_engine.clear_history()
            st.rerun()
    
    repo_filter = None if selected_repo == "All Repositories" else selected_repo
    
    # Chat interface
    st.divider()
    
    # Display conversation history
    for i, exchange in enumerate(st.session_state.conversation_history):
        with st.chat_message("user"):
            st.write(exchange['question'])
        
        with st.chat_message("assistant"):
            st.write(exchange['answer'])
            
            # Show sources
            if exchange.get('sources'):
                with st.expander("📚 View Sources"):
                    for source in exchange['sources']:
                        st.code(f"File: {source['file_path']}\n"
                               f"Type: {source['chunk_type']}\n"
                               f"Lines: {source['start_line']}-{source['end_line']}\n"
                               f"Score: {source['score']:.3f}")
    
    # Query input
    query = st.chat_input("Ask a question about the code...")
    
    if query:
        # Add user message
        with st.chat_message("user"):
            st.write(query)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Searching and analyzing code..."):
                try:
                    result = st.session_state.query_engine.query(
                        question=query,
                        max_results=st.session_state.get('top_k', 5),
                        repository_filter=repo_filter,
                        use_codebert=st.session_state.get('use_codebert', False)
                    )
                    
                    st.write(result.answer)
                    
                    # Show confidence and sources
                    st.caption(f"Confidence: {result.confidence:.2%}")
                    
                    # Handle both old and new result formats
                    sources_to_display = []
                    if hasattr(result, 'final_code_blocks'):
                        for block in result.final_code_blocks:
                            sources_to_display.append({
                                'metadata': {
                                    'file_path': block['file_path'],
                                    'chunk_type': block['chunk_type'],
                                    'name': block['function_name'],
                                    'start_line': block['start_line'],
                                    'end_line': block['end_line']
                                },
                                'score': block['score']
                            })
                    elif hasattr(result, 'sources'):
                        sources_to_display = result.sources
                    
                    if sources_to_display:
                        with st.expander("📚 View Sources"):
                            for source in sources_to_display:
                                meta = source['metadata']
                                st.code(f"File: {meta.get('file_path', 'unknown')}\n"
                                       f"Type: {meta.get('chunk_type', 'code')}\n"
                                       f"Name: {meta.get('name', 'N/A')}\n"
                                       f"Lines: {meta.get('start_line', '?')}-{meta.get('end_line', '?')}\n"
                                       f"Score: {source['score']:.3f}")
                    
                    # Store in history
                    st.session_state.conversation_history.append({
                        'question': query,
                        'answer': result.answer,
                        'sources': [
                            {
                                'file_path': s['metadata'].get('file_path', 'unknown'),
                                'chunk_type': s['metadata'].get('chunk_type', 'code'),
                                'start_line': s['metadata'].get('start_line', 0),
                                'end_line': s['metadata'].get('end_line', 0),
                                'score': s['score']
                            }
                            for s in sources_to_display
                        ],
                        'confidence': result.confidence
                    })
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    logger.exception("Query processing error")

def render_browse_interface():
    """Render file browsing interface"""
    st.header("📂 Browse Indexed Files")
    
    components = get_components()
    vector_store = components['vector_store']
    
    repos = vector_store.store.get_all_repos()
    
    if not repos:
        st.info("No repositories indexed yet")
        return
    
    selected_repo = st.selectbox("Select Repository", repos)
    
    if selected_repo:
        files = vector_store.store.get_files_for_repo(selected_repo)
        
        if files:
            st.write(f"**Files:** {len(files)}")
            
            # Group by directory
            directories = {}
            for f in files:
                parts = f.split('/')
                if len(parts) > 1:
                    dir_name = parts[0]
                else:
                    dir_name = "/"
                
                if dir_name not in directories:
                    directories[dir_name] = []
                directories[dir_name].append(f)
            
            # Display file tree
            for dir_name, dir_files in sorted(directories.items()):
                with st.expander(f"📁 {dir_name} ({len(dir_files)} files)"):
                    for f in sorted(dir_files):
                        st.text(f"  📄 {f}")
        else:
            st.info("No files found")


def main():
    """Main application entry point"""
    init_session_state()
    
    # Title and description
    st.title("🔍 Git Repository Q&A")
    st.markdown("""
    **Intelligent code repository question answering powered by semantic search and LLM.**
    
    Index your Git repositories and ask natural language questions about the code!
    """)
    
    # Initialize components
    try:
        get_components()
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.info("Please check your configuration and environment variables")
        return
    
    # Render sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📥 Index", "💬 Ask", "📂 Browse"])
    
    with tab1:
        render_repo_input()
    
    with tab2:
        render_qa_interface()
    
    with tab3:
        render_browse_interface()
    
    # Footer
    st.divider()
    st.caption("Git Repository Q&A POC | Powered by Azure OpenAI, ChromaDB, and LangChain")


if __name__ == "__main__":
    main()
