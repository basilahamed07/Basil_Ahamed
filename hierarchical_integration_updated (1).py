"""
Updated Integration Module for Hierarchical Retrieval System
Integrates both ChromaDB and OpenSearch for hybrid retrieval
"""
import logging
from typing import Optional, Dict, List

from config import Config
from embeddings import EmbeddingManager
from vector_store import VectorStoreManager, ChromaDBStore
from query_processor import CodeQAProcessor
from opensearch_handler_new import OpenSearchHandler
from hierarchical_retriever import HierarchicalRetriever, ModulePathExtractor
from hierarchical_query_engine import HierarchicalQueryEngine, ConversationalHierarchicalEngine
from hybrid_retriever import HybridRetriever, HybridSearchConfig, HybridChromaDBStore

logger = logging.getLogger(__name__)


class HybridVectorStoreManager:
    """
    Extended VectorStoreManager that includes OpenSearch for hybrid search
    """
    
    def __init__(self, 
                 vector_store_manager: VectorStoreManager,
                 opensearch_handler: OpenSearchHandler,
                 hybrid_config: Optional[HybridSearchConfig] = None):
        """
        Wrap VectorStoreManager with hybrid capabilities
        
        Args:
            vector_store_manager: Original VectorStoreManager
            opensearch_handler: OpenSearchHandler for keyword search
            hybrid_config: Configuration for hybrid search weights
        """
        self._vector_store = vector_store_manager
        self._opensearch = opensearch_handler
        self._hybrid_config = hybrid_config or HybridSearchConfig()
        self._last_query_text = None
        
        # Create hybrid store wrapper if we have a base store
        self._hybrid_store = None
        
    @property
    def store(self):
        """Get the hybrid store (used by hierarchical retriever)"""
        if self._hybrid_store is None and self._vector_store.store is not None:
            self._hybrid_store = HybridChromaDBStore(
                chroma_store=self._vector_store.store,
                opensearch_handler=self._opensearch,
                hybrid_config=self._hybrid_config
            )
        return self._hybrid_store or self._vector_store.store
    
    def set_query_text(self, query_text: str):
        """Set the query text for hybrid search"""
        self._last_query_text = query_text
        if self._hybrid_store:
            self._hybrid_store.set_query_text(query_text)
        # Also set on any store that gets created later
        if self.store:
            self.store.set_query_text(query_text)
    
    def semantic_search(self, 
                        query_embedding: List[float],
                        top_k: int = 5,
                        repo_filter: Optional[str] = None,
                        language_filter: Optional[str] = None,
                        file_filter: Optional[str] = None) -> List[Dict]:
        """
        Perform hybrid semantic search
        """
        # Get more results for fusion
        fetch_k = top_k * 3
        
        # Get ChromaDB results
        chroma_results = self._vector_store.semantic_search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            repo_filter=repo_filter,
            language_filter=language_filter,
            file_filter=file_filter
        )
        
        # Mark source
        for r in chroma_results:
            r['retrieval_source'] = 'chroma_semantic'
        
        # Get OpenSearch results if we have query text
        if self._last_query_text and self._opensearch:
            try:
                opensearch_results = self._search_opensearch(
                    query_text=self._last_query_text,
                    top_k=fetch_k,
                    repo_filter=repo_filter,
                    language_filter=language_filter
                )
                
                # Combine using RRF
                combined = self._rrf_fusion(chroma_results, opensearch_results)
                combined.sort(key=lambda x: x['score'], reverse=True)
                return combined[:top_k]
                
            except Exception as e:
                logger.warning(f"OpenSearch failed, using ChromaDB only: {e}")
        
        return chroma_results[:top_k]
    
    def _search_opensearch(self,
                           query_text: str,
                           top_k: int,
                           repo_filter: Optional[str] = None,
                           language_filter: Optional[str] = None) -> List[Dict]:
        """Search OpenSearch and format results"""
        try:
            if repo_filter:
                self._opensearch.set_repo_context(repo_filter)
            
            results = self._opensearch.search_chunks(
                query=query_text,
                repo_name=repo_filter,
                language=language_filter,
                size=top_k
            )
            
            # Format to match ChromaDB results
            formatted = []
            for r in results:
                raw_score = r.get('score', 0) or 0
                normalized_score = min(raw_score / 10.0, 1.0) if raw_score > 0 else 0.0
                
                formatted.append({
                    'id': r.get('chunk_id', ''),
                    'content': r.get('content', r.get('original_content', '')),
                    'metadata': {
                        'file_path': r.get('file_path', ''),
                        'repo_name': r.get('repo_name', ''),
                        'language': r.get('language', ''),
                        'chunk_type': r.get('chunk_type', ''),
                        'name': r.get('name', ''),
                        'start_line': r.get('start_line', 0),
                        'end_line': r.get('end_line', 0),
                        'parent_class': r.get('parent_class', ''),
                        'docstring': r.get('docstring', ''),
                        'file_summary': r.get('file_summary', '')
                    },
                    'score': normalized_score,
                    'retrieval_source': 'opensearch_keyword'
                })
            
            return formatted
            
        except Exception as e:
            logger.error(f"OpenSearch search error: {e}")
            return []
    
    def _rrf_fusion(self, 
                    chroma_results: List[Dict],
                    opensearch_results: List[Dict]) -> List[Dict]:
        """Combine results using Reciprocal Rank Fusion"""
        k = self._hybrid_config.rrf_k
        result_map = {}
        
        # Process ChromaDB results
        for rank, result in enumerate(chroma_results, 1):
            key = self._get_result_key(result)
            if key not in result_map:
                result_map[key] = {
                    'data': result,
                    'score': 0.0,
                    'sources': []
                }
            result_map[key]['score'] += self._hybrid_config.chroma_weight / (k + rank)
            result_map[key]['sources'].append('chroma_semantic')
        
        # Process OpenSearch results
        for rank, result in enumerate(opensearch_results, 1):
            key = self._get_result_key(result)
            if key not in result_map:
                result_map[key] = {
                    'data': result,
                    'score': 0.0,
                    'sources': []
                }
            result_map[key]['score'] += self._hybrid_config.opensearch_weight / (k + rank)
            result_map[key]['sources'].append('opensearch_keyword')
        
        # Build final results
        combined = []
        for data in result_map.values():
            result = data['data'].copy()
            result['score'] = data['score']
            result['retrieval_sources'] = list(set(data['sources']))
            result['is_hybrid_match'] = len(data['sources']) > 1
            combined.append(result)
        
        return combined
    
    def _get_result_key(self, result: Dict) -> str:
        """Generate unique key for deduplication"""
        metadata = result.get('metadata', {})
        file_path = metadata.get('file_path', '')
        start_line = str(metadata.get('start_line', 0))
        name = metadata.get('name', '')
        return f"{file_path}::{start_line}::{name}"
    
    # Delegate other methods to original vector store
    def __getattr__(self, name):
        return getattr(self._vector_store, name)


class HybridHierarchicalQueryEngine(HierarchicalQueryEngine):
    """
    Extended Hierarchical Query Engine with hybrid retrieval
    """
    
    def __init__(self, 
                 embedding_manager: EmbeddingManager,
                 vector_store_manager: VectorStoreManager,
                 opensearch_handler: OpenSearchHandler,
                 llm_processor: CodeQAProcessor,
                 hybrid_config: Optional[HybridSearchConfig] = None):
        """
        Initialize with hybrid retrieval capability
        
        Args:
            embedding_manager: For query embedding
            vector_store_manager: ChromaDB vector store
            opensearch_handler: OpenSearch for keyword search  
            llm_processor: LLM for answer generation
            hybrid_config: Configuration for hybrid weights
        """
        self.embedding_manager = embedding_manager
        self.opensearch = opensearch_handler
        self.llm = llm_processor
        self._hybrid_config = hybrid_config or HybridSearchConfig()
        
        # Create hybrid vector store wrapper
        self.hybrid_vector_store = HybridVectorStoreManager(
            vector_store_manager=vector_store_manager,
            opensearch_handler=opensearch_handler,
            hybrid_config=self._hybrid_config
        )
        
        # Use hybrid store for hierarchical retriever
        self.vector_store = self.hybrid_vector_store
        
        # Import answer builder
        from hierarchical_query_engine import HierarchicalAnswerBuilder
        
        self.hierarchical_retriever = HierarchicalRetriever(self.hybrid_vector_store)
        self.answer_builder = HierarchicalAnswerBuilder(llm_processor)
        
        logger.info("HybridHierarchicalQueryEngine initialized with OpenSearch integration")
    
    def query(self, question: str,
              repository_filter: Optional[str] = None,
              module_filter: Optional[str] = None,
              use_codebert: bool = False,
              max_results: int = 10):
        """
        Process query with hybrid hierarchical retrieval
        """
        from hierarchical_query_engine import EnhancedQueryResult
        
        logger.info(f"Processing hybrid query: {question}")
        
        # Set query text for hybrid search
        self.hybrid_vector_store.set_query_text(question)
        
        # Check for overview questions
        overview_keywords = ['explain', 'overview', 'project', 'application', 'describe', 
                            'architecture', 'structure', 'what is this', 'what does',
                            'summary', 'summarize', 'tell me about', 'how does this work']
        is_overview = any(keyword in question.lower() for keyword in overview_keywords)
        
        if is_overview:
            max_results = min(max_results * 2, 20)
            logger.info("Detected overview question, increasing retrieval scope")
        
        # Embed query
        query_embedding = self.embedding_manager.embed_query(
            question,
            use_codebert=use_codebert
        )
        
        # Perform hierarchical retrieval (now using hybrid store)
        retrieval_result = self.hierarchical_retriever.retrieve(
            query_embedding,
            repository_filter=repository_filter,
            module_filter=module_filter,
            max_final_results=max_results
        )
        
        if not retrieval_result.final_chunks:
            return EnhancedQueryResult(
                answer="No relevant code found in the repository to answer this question.",
                final_code_blocks=[],
                hierarchy_context={},
                retrieval_stats={},
                confidence=0.0,
                query=question
            )
        
        # Generate answer
        answer = self.answer_builder.generate_answer(question, retrieval_result)
        
        # Get stats
        retrieval_stats = self.hierarchical_retriever.get_retrieval_statistics(retrieval_result)
        
        # Add hybrid retrieval info to stats
        hybrid_matches = sum(1 for c in retrieval_result.final_chunks 
                           if c.get('is_hybrid_match', False))
        retrieval_stats['hybrid_matches'] = hybrid_matches
        retrieval_stats['retrieval_mode'] = 'hybrid'
        
        avg_score = retrieval_stats.get('avg_score', 0.0)
        
        return EnhancedQueryResult(
            answer=answer,
            final_code_blocks=retrieval_result.final_chunks,
            hierarchy_context=retrieval_result.hierarchy_context,
            retrieval_stats=retrieval_stats,
            confidence=float(avg_score),
            query=question
        )


class HybridConversationalEngine(HybridHierarchicalQueryEngine):
    """
    Conversational engine with hybrid retrieval and history support
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history = []
        self.max_history = 10
    
    def query_with_history(self, question: str, **kwargs):
        """Process query with conversation history"""
        from hierarchical_query_engine import EnhancedQueryResult
        
        enhanced_question = question
        
        if self.conversation_history:
            recent_history = self.conversation_history[-2:]
            history_summary = []
            
            for h in recent_history:
                history_summary.append(f"Previous Q: {h['question'][:100]}")
                history_summary.append(f"Previous A: {h['answer'][:150]}...")
            
            if history_summary:
                context_prefix = "Context from previous conversation:\n"
                context_prefix += "\n".join(history_summary)
                context_prefix += "\n\nCurrent question: "
                enhanced_question = context_prefix + question
        
        result = self.query(enhanced_question, **kwargs)
        
        self.conversation_history.append({
            'question': question,
            'answer': result.answer,
            'files': [b['file_path'] for b in result.final_code_blocks[:5]],
            'hierarchy_path': result.hierarchy_context.get('hierarchy_path', {})
        })
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        return result
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history


class HybridSystemInitializer:
    """
    Initialize the complete hybrid retrieval system
    """
    
    @staticmethod
    def initialize_from_config(config: Config,
                               opensearch_handler: Optional[OpenSearchHandler] = None,
                               hybrid_config: Optional[HybridSearchConfig] = None) -> Dict:
        """
        Initialize all components with hybrid retrieval
        
        Args:
            config: Application configuration
            opensearch_handler: Optional pre-initialized OpenSearch handler
            hybrid_config: Optional hybrid search configuration
            
        Returns:
            Dictionary containing all initialized components
        """
        logger.info("Initializing hybrid hierarchical retrieval system")
        
        embedding_manager = EmbeddingManager(config)
        
        vector_store = VectorStoreManager(
            persist_directory=config.chroma.persist_directory,
            base_collection_name=config.chroma.collection_name
        )
        
        llm_processor = CodeQAProcessor(
            azure_endpoint=config.azure_openai.endpoint,
            azure_api_key=config.azure_openai.api_key,
            deployment_name=config.azure_openai.chat_deployment,
            api_version=config.azure_openai.api_version
        )
        
        # Initialize OpenSearch if not provided
        if opensearch_handler is None:
            opensearch_handler = OpenSearchHandler(
                host=config.opensearch.host,
                port=config.opensearch.port,
                use_ssl=config.opensearch.use_ssl,
                verify_certs=config.opensearch.verify_certs,
                index_name=config.opensearch.index_name
            )
        
        # Use default hybrid config if not provided
        if hybrid_config is None:
            hybrid_config = HybridSearchConfig(
                chroma_weight=0.6,
                opensearch_weight=0.4,
                rrf_k=60,
                use_rrf=True
            )
        
        # Create hybrid query engine
        hybrid_engine = HybridHierarchicalQueryEngine(
            embedding_manager=embedding_manager,
            vector_store_manager=vector_store,
            opensearch_handler=opensearch_handler,
            llm_processor=llm_processor,
            hybrid_config=hybrid_config
        )
        
        # Create conversational engine
        conversational_engine = HybridConversationalEngine(
            embedding_manager=embedding_manager,
            vector_store_manager=vector_store,
            opensearch_handler=opensearch_handler,
            llm_processor=llm_processor,
            hybrid_config=hybrid_config
        )
        
        # Also create standard components for backward compatibility
        hierarchical_retriever = HierarchicalRetriever(vector_store)
        
        logger.info("Hybrid hierarchical retrieval system initialized successfully")
        
        return {
            'config': config,
            'embedding_manager': embedding_manager,
            'vector_store': vector_store,
            'opensearch_handler': opensearch_handler,
            'llm_processor': llm_processor,
            'hierarchical_retriever': hierarchical_retriever,
            'hierarchical_engine': hybrid_engine,  # Now uses hybrid
            'conversational_engine': conversational_engine,  # Now uses hybrid
            'hybrid_config': hybrid_config
        }


# Convenience function for quick setup
def get_hybrid_components(config: Optional[Config] = None,
                          hybrid_config: Optional[HybridSearchConfig] = None):
    """
    Convenience function to get hybrid components
    
    Usage:
        components = get_hybrid_components()
        engine = components['conversational_engine']
        result = engine.query("How does authentication work?")
    """
    if config is None:
        from config import config as default_config
        config = default_config
    
    return HybridSystemInitializer.initialize_from_config(config, hybrid_config=hybrid_config)
