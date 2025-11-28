# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "duckdb==1.3.2",
#     "ipython[black,doc]==9.4.0",
#     "langchain-community==0.3.27",
#     "langchain[community,ollama]==0.3.27",
#     "marimo",
#     "numpy==2.3.2",
#     "openai==1.98.0",
#     "sentence-transformers[onnx,onnx-gpu,openvino]==5.1.0",
#     "torch==2.8.0",
#     "transformers[hf-xet]==4.55.0",
# ]
# ///

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium", sql_output="native")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from typing import List, Dict, Any
    from langchain.chat_models import ChatOpenAI
    return Any, ChatOpenAI, Dict, List, SentenceTransformer, duckdb, mo, np


@app.cell
def _(ChatOpenAI):
    llm = ChatOpenAI(
        model="ai/smollm3",
        openai_api_key="ollma",
        base_url="http://localhost:12434/engines/v1",
    )
    return (llm,)


@app.cell(hide_code=True)
def _(SentenceTransformer, duckdb, mo, np):
    # Cell 3: DuckDB VSS Extension Setup (Correct Configuration)
    @mo.cache
    def setup_duckdb_vss():
        """Configure DuckDB with VSS Extension and create vector index"""

        # DuckDB connection (persistent)
        con = duckdb.connect("rag_production.duckdb")

        # Install and load VSS Extension
        con.execute("INSTALL vss;")
        con.execute("LOAD vss;")

        # Enable experimental persistence (important for production environment)
        con.execute("SET hnsw_enable_experimental_persistence = true;")

        # Load embedding model
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_dim = 384

        # Create documents table
        con.execute("DROP TABLE IF EXISTS documents;")
        con.execute(f"""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                title TEXT,
                metadata TEXT,
                embedding FLOAT[{embedding_dim}],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Comprehensive knowledge base for RAG
        knowledge_base = [
            (
                "Retrieval-Augmented Generation (RAG) combines the strengths of information retrieval and generative AI. RAG systems use vector databases for semantic search and LLMs for answer generation.",
                "RAG Basics",
                "rag,llm,vector_search",
            ),
            (
                "DuckDB VSS Extension provides HNSW indexing for ultra-fast vector search. The extension has been available since DuckDB 1.0.0 and is production-ready for RAG applications.",
                "DuckDB VSS",
                "duckdb,hnsw,vector_search",
            ),
            (
                "HNSW (Hierarchical Navigable Small Worlds) is an algorithm for Approximate Nearest Neighbor Search. It offers logarithmic search times and high recall rates.",
                "HNSW Algorithm",
                "hnsw,algorithm,performance",
            ),
            (
                "marimo is a reactive Python notebook with integrated UI components. mo.ui.chat enables simple chatbot development with Python functions.",
                "marimo Framework",
                "marimo,notebook,ui",
            ),
            (
                "Embedding models convert text into numerical vectors for semantic search. all-MiniLM-L6-v2 is a proven model for German and English texts.",
                "Embedding Models",
                "embeddings,nlp,semantic_search",
            ),
            (
                "Production RAG requires monitoring, evaluation, and robust error handling. Performance metrics such as retrieval time and answer quality are crucial.",
                "RAG Production",
                "production,monitoring,evaluation",
            ),
            (
                "Hybrid search strategies combine vector search with keyword matching for better results. This approach improves both precision and recall.",
                "Hybrid Search",
                "hybrid,search,precision",
            ),
            (
                "Chunking strategies significantly influence RAG quality. Optimal chunk sizes are between 256-1024 tokens with 10-20% overlap.",
                "Text Chunking",
                "chunking,preprocessing,optimization",
            ),
            (
                "Local LLMs like SmolLM3 or Llama 3.2 enable privacy-compliant RAG systems. Ollama simplifies local LLM deployment.",
                "Local LLMs",
                "local_llm,privacy,ollama",
            ),
            (
                "Vector indexes require regular maintenance. DuckDB VSS provides PRAGMA functions for index compaction and optimization.",
                "Index Maintenance",
                "maintenance,optimization,performance",
            ),
            (
                "Semantic search finds conceptually similar content through vector similarity. Cosine similarity and Euclidean distance are commonly used metrics.",
                "Semantic Search",
                "semantic,similarity,metrics",
            ),
            (
                "Multi-Modal RAG processes text, images, and videos. Vision-Language models enable the integration of visual content into RAG systems.",
                "Multi-Modal RAG",
                "multimodal,vision,language",
            ),
            (
                "Enterprise RAG requires governance, compliance, and auditability. Bias detection and explainability are important requirements.",
                "Enterprise RAG",
                "enterprise,governance,compliance",
            ),
            (
                "GraphRAG uses knowledge graphs for complex relational queries. Particularly effective for structured knowledge domains.",
                "GraphRAG",
                "graph,knowledge,relations",
            ),
            (
                "RAG evaluation uses metrics such as Faithfulness, Answer Relevancy, and Context Recall. RAGAS is an established framework for RAG assessment.",
                "RAG Evaluation",
                "evaluation,metrics,ragas",
            ),
        ]

        # Compute and store embeddings
        for i, (content, title, metadata) in enumerate(knowledge_base):
            embedding = embedder.encode([content])[0]
            con.execute(
                """
                INSERT INTO documents (id, content, title, metadata, embedding)
                VALUES (?, ?, ?, ?, ?)
            """,
                (i, content, title, metadata, embedding.astype(np.float32)),
            )

        # Create HNSW index (after data insert for optimal performance)
        con.execute("CREATE INDEX doc_hnsw_idx ON documents USING HNSW(embedding);")

        return con, embedder


    # Initialize database and embedder
    db, embedder = setup_duckdb_vss()

    # Display statistics
    stats = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    mo.md(f"🦆 **DuckDB VSS configured: {stats} documents with HNSW index**")
    return db, embedder


@app.cell
def _(Any, ChatMessage, Dict, List, db, embedder, llm, mo, np, query):
    # Cell 5: Complete RAG Pipeline with DuckDB VSS
    class RAG:
        """Production-ready RAG Pipeline with DuckDB VSS"""

        def __init__(self, db_connection, embedder, llm):
            self.db = db_connection
            self.embedder = embedder
            self.llm = llm

        def retrieve_documents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
            """Semantic search with DuckDB VSS"""
            try:
                # Create query embedding
                query_embedding = self.embedder.encode([query])[0]

                # Vector search with HNSW index
                results = self.db.execute(
                    """
                    SELECT 
                        id,
                        content,
                        title,
                        metadata,
                        array_cosine_distance(embedding, ?::FLOAT[384]) as distance
                    FROM documents
                    ORDER BY distance ASC
                    LIMIT ?
                """,
                    (query_embedding.astype(np.float32), k),
                ).fetchall()

                # Format results
                retrieved_docs = []
                for id_, content, title, metadata, distance in results:
                    retrieved_docs.append({
                        "id": id_,
                        "content": content,
                        "title": title,
                        "metadata": metadata,
                        "similarity": 1 - distance,  # Cosine similarity
                        "distance": distance,
                    })

                return retrieved_docs

            except Exception as e:
                return [{"error": f"Retrieval error: {str(e)}"}]

        def generate_response(self, message: ChatMessage) -> str:
            """Complete RAG Pipeline: Retrieve + Generate"""

            # 1. Document search
            retrieved_docs = self.retrieve_documents(message.content)

            if not retrieved_docs or "error" in retrieved_docs[0]:
                return "No relevant documents found."

            # 2. Compile context
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                context_parts.append(
                    f"Document {i}: {doc['title']}\n"
                    f"Content: {doc['content']}\n"
                    f"Similarity: {doc['similarity']:.3f}\n"
                )

            context = "\n".join(context_parts)

            # 3. Create RAG prompt
            prompt = f"""You are a helpful AI assistant with access to a knowledge base.

    CONTEXT from the knowledge base:
    {context}

    USER QUESTION: {query}

    INSTRUCTIONS:
    - Answer the question based on the provided context
    - Cite relevant sources with similarity values
    - If the context is insufficient, say so honestly
    - Answer in English and structured

    ANSWER:"""

            # 4. LLM generation
            response = self.llm.generate(prompt)

            # 5. Add metadata
            metadata = f"\n\n📚 **Sources:** {len(retrieved_docs)} documents, best similarity: {retrieved_docs[0]['similarity']:.3f}"

            return response + metadata


    # Initialize RAG pipeline
    rag = RAG(db, embedder, llm)

    mo.md("🚀 **Production-ready RAG Pipeline ready**")
    return (rag,)


@app.cell
def _(List, mo, rag):
    # Cell 6: Correct marimo Chat Implementation
    def rag_chat_model(messages: List, config=None) -> str:
        """
        Correct marimo Chat implementation for RAG

        IMPORTANT: messages is a list of ChatMessage objects
        Access content via message.content (not message['content'])
        """

        # Safety check
        if not messages:
            return "No messages received."

        # Extract last message (correct attribute syntax)
        last_message = messages[-1]
        user_query = last_message

        # Only process user messages
        if last_message.role != "user":
            return "Only user messages are processed."

        # Execute RAG pipeline
        try:
            response = rag.generate_response(user_query)
            return response

        except Exception as e:
            return f"🚨 RAG Pipeline error: {str(e)}"


    mo.md("💬 **Correct marimo Chat function implemented**")
    return (rag_chat_model,)


@app.cell
def _(mo, rag_chat_model):
    chat = mo.ui.chat(
        model=rag_chat_model,
        prompts=["Write a haiku about recursion in programming."],
    )
    chat
    return


if __name__ == "__main__":
    app.run()
