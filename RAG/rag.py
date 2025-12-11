# ============================================================================
# PART 1: CORE RAG ENGINE
# ============================================================================

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
import uuid
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
import re
from datetime import datetime

load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration"""
    CSV_PATH = "V:\RAG\data\710edited.csv"
    VECTOR_DB_PATH = "./chroma_legal_db"
    
    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "llama-3.1-8b-instant"
    
    # Chunking parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval parameters
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.3
    
    # LLM parameters
    TEMPERATURE = 0.1
    MAX_TOKENS = 1024

print("Configuration loaded!")


# ============================================================================
# EMBEDDING MANAGER
# ============================================================================

class EmbeddingManager:
    """Handles all embedding operations"""
    
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load SentenceTransformer model"""
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"Model loaded! Dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.model.encode(text, show_progress_bar=False)
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        return self.model.encode(texts, show_progress_bar=True, batch_size=32)


# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """Process and chunk legal documents"""
    
    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, 
                 chunk_overlap: int = Config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def clean_text(self, text: str) -> str:
        """Clean legal text"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove citations (optional - keep if you want citations in context)
        # text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)
        
        return text.strip()
    
    def process_case(self, case_id: str, title: str, content: str) -> List[Document]:
        """Process a single case into chunks"""
        
        # Clean content
        clean_content = self.clean_text(content)
        
        # Create base document
        base_doc = Document(
            page_content=clean_content,
            metadata={
                'case_id': str(case_id),
                'title': title,
                'content_length': len(clean_content),
                'document_type': 'legal_case'
            }
        )
        
        # Split into chunks
        chunks = self.text_splitter.split_documents([base_doc])
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['total_chunks'] = len(chunks)
            chunk.metadata['chunk_start_char'] = i * (self.chunk_size - self.chunk_overlap)
        
        return chunks
    
    def process_csv(self, csv_path: str) -> Tuple[List[Document], pd.DataFrame]:
        """Process entire CSV file"""
        print(f"Loading CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Found {len(df)} cases")
        
        all_chunks = []
        
        for idx, row in df.iterrows():
            case_id = str(row['id'])
            title = str(row['title'])
            content = str(row['content'])
            
            chunks = self.process_case(case_id, title, content)
            all_chunks.extend(chunks)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)} cases...")
        
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks, df


# ============================================================================
# VECTOR STORE MANAGER
# ============================================================================

class VectorStoreManager:
    """Manages ChromaDB vector store operations"""
    
    def __init__(self, persist_directory: str = Config.VECTOR_DB_PATH, collection_name: str = "legal_cases"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB"""
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Try to get existing collection, or create new one
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
            print(f"Documents in collection: {self.collection.count()}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Legal case embeddings for RAG"}
            )
            print(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray, batch_size: int = 1000):
        """Add documents to vector store"""
        
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings count mismatch!")
        
        print(f"Adding {len(documents)} documents in batches of {batch_size}...")
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            ids = []
            metadatas = []
            contents = []
            embeddings_list = []
            
            for j, (doc, emb) in enumerate(zip(batch_docs, batch_embeddings)):
                doc_id = f"{doc.metadata['case_id']}_chunk_{doc.metadata['chunk_id']}"
                
                ids.append(doc_id)
                metadatas.append(doc.metadata)
                contents.append(doc.page_content)
                embeddings_list.append(emb.tolist())
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=contents
            )
            
            print(f"    Batch {i//batch_size + 1}: Added {len(batch_docs)} documents")
        
        print(f"Total documents in collection: {self.collection.count()}")
    
    def search_by_case(self, query_embedding: np.ndarray, case_id: str, top_k: int = Config.TOP_K) -> List[Dict]:
        """Search within a specific case"""
        
        # Query with case_id filter
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 3,  # Get more results to filter
            where={"case_id": str(case_id)}
        )
        
        # Process results
        retrieved_docs = []
        
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                # Calculate similarity score from distance
                distance = results['distances'][0][i]
                similarity = 1 - (distance / 2)  # Convert L2 distance to similarity
                
                retrieved_docs.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': similarity,
                    'chunk_id': results['metadatas'][0][i].get('chunk_id', 0)
                })
        
        # Sort by similarity and take top_k
        retrieved_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
        return retrieved_docs[:top_k]


# ============================================================================
# LLM MANAGER
# ============================================================================

class LLMManager:
    """Manages LLM interactions with Groq"""
    
    def __init__(self, model_name: str = Config.LLM_MODEL):
        self.model_name = model_name
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables!")
        
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        
        print(f"LLM initialized: {model_name}")
    
    def generate_answer(self, question: str, context: str, case_title: str) -> str:
        """Generate answer using retrieved context"""
        
        prompt = f"""You are an expert legal AI assistant analyzing Indian Supreme Court cases.

Case Title: {case_title}

Context from the case:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. Be precise and cite specific parts of the judgment
3. Use legal terminology appropriately
4. If the context doesn't contain enough information, say so clearly
5. Keep your answer concise but comprehensive

Answer:"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"


# ============================================================================
# RAG ENGINE (MAIN CLASS)
# ============================================================================

class LegalRAGEngine:
    """Complete RAG pipeline for legal cases"""
    
    def __init__(self):
        print("\n" + "="*80)
        print("🚀 INITIALIZING LEGAL RAG ENGINE")
        print("="*80 + "\n")
        
        self.embedding_manager = EmbeddingManager()
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStoreManager()
        self.llm_manager = LLMManager()
        self.case_metadata = {}  # Store case metadata for quick access
        
        print("\n✅ RAG Engine initialized successfully!\n")
    
    def index_cases(self, csv_path: str = Config.CSV_PATH, force_reindex: bool = False):
        """Index all cases from CSV"""
        
        # Check if already indexed
        if self.vector_store.collection.count() > 0 and not force_reindex:
            print("    Vector store already contains documents!")
            print("   Set force_reindex=True to re-index")
            
            # Load case metadata
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.case_metadata[str(row['id'])] = {
                    'title': str(row['title']),
                    'id': str(row['id'])
                }
            return
        
        # Process documents
        chunks, df = self.doc_processor.process_csv(csv_path)
        
        # Store case metadata
        for _, row in df.iterrows():
            self.case_metadata[str(row['id'])] = {
                'title': str(row['title']),
                'id': str(row['id'])
            }
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_manager.generate_embeddings_batch(texts)
        
        # Store in vector database
        print("\nStoring in vector database...")
        self.vector_store.add_documents(chunks, embeddings)
        
        print("\nIndexing complete!")
    
    def query(self, case_id: str, question: str, top_k: int = Config.TOP_K) -> Dict:
        """Query the RAG system for a specific case"""
        
        start_time = datetime.now()
        
        # Validate case_id
        case_id = str(case_id)
        if case_id not in self.case_metadata:
            return {
                'success': False,
                'error': f'Case ID {case_id} not found in database',
                'answer': None
            }
        
        case_title = self.case_metadata[case_id]['title']
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embedding(question)
        
        # Retrieve relevant chunks
        retrieved_docs = self.vector_store.search_by_case(
            query_embedding, 
            case_id, 
            top_k=top_k
        )
        
        if not retrieved_docs:
            return {
                'success': False,
                'error': 'No relevant content found for this question',
                'answer': None,
                'case_title': case_title
            }
        
        # Filter by similarity threshold
        relevant_docs = [
            doc for doc in retrieved_docs 
            if doc['similarity_score'] >= Config.SIMILARITY_THRESHOLD
        ]
        
        if not relevant_docs:
            return {
                'success': False,
                'error': 'Question not relevant to this case',
                'answer': None,
                'case_title': case_title,
                'suggestion': 'Try rephrasing your question or ask about specific aspects of the case'
            }
        
        # Prepare context
        context = "\n\n---\n\n".join([doc['content'] for doc in relevant_docs])
        
        # Generate answer
        answer = self.llm_manager.generate_answer(question, context, case_title)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'success': True,
            'answer': answer,
            'case_id': case_id,
            'case_title': case_title,
            'question': question,
            'retrieved_chunks': len(relevant_docs),
            'avg_similarity': sum(d['similarity_score'] for d in relevant_docs) / len(relevant_docs),
            'processing_time_seconds': processing_time,
            'sources': [
                {
                    'chunk_id': doc['chunk_id'],
                    'similarity': round(doc['similarity_score'], 3),
                    'preview': doc['content'][:200] + "..."
                }
                for doc in relevant_docs
            ]
        }

print("All classes loaded successfully!")
print("\n" + "="*80)

