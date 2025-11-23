import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium", sql_output="native")


@app.cell
def _():
    import marimo as mo
    from langchain.agents import AgentType, initialize_agent
    return AgentType, initialize_agent, mo


@app.cell
def _(mo):
    from typing import Any, List, Optional

    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.llms.base import LLM
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Cell 5: SmolLM3 as LangChain LLM Wrapper
    class SmolLM3LLM(LLM):
        """SmolLM3 as LangChain-compatible LLM"""
    
        def __init__(self):
            super().__init__()
            self.model_name = "HuggingFaceTB/SmolLM3-3B"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        @property
        def _llm_type(self) -> str:
            return "smollm3"
    
        def _call(self, prompt: str, stop: Optional[List[str]] = None, 
                 run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
            response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            return response.strip()

    # Initialize LLM
    smollm = SmolLM3LLM()
    mo.md("🤖 **SmolLM3 successfully loaded**")

    return


@app.cell
def _():
    from sentence_transformers import SentenceTransformer

    texts = [
        "Ketanji Brown Jackson is a justice on the Supreme Court of the United States.",
        "The president praised Ketanji Brown Jackson for her integrity.",
        "The weather is sunny and warm today.",
    ]
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts)
    return (embedder,)


@app.cell
def _():
    import duckdb
    import numpy as np
    return duckdb, np


@app.cell
def _(duckdb, mo):
    # Cell 6: Setup DuckDB with VSS Extension
    # DuckDB connection with VSS Extension
    con = duckdb.connect("rag_agent.duckdb")

    # Install and load VSS Extension
    con.execute("INSTALL vss;")
    con.execute("LOAD vss;")
    con.execute("SET hnsw_enable_experimental_persistence = true;")

    mo.md("🦆 **DuckDB with VSS Extension successfully configured**")

    return (con,)


@app.cell
def _(con, mo):
    # Create table for documents
    con.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            content TEXT,
            metadata TEXT,
            embedding FLOAT[384]
        )
    """)

    # Create HNSW index (384 is the dimension of all-MiniLM-L6-v2)
    con.execute("""
        CREATE INDEX IF NOT EXISTS doc_embedding_idx 
        ON documents USING HNSW (embedding) 
        WITH (metric = 'cosine')
    """)
    mo.md("🦆 **DuckDB tables successfully created**")
    return


@app.cell
def _():
    sample_docs = [
        ("Artificial Intelligence (AI) is the simulation of human intelligence in machines.", "topic: AI"),
        ("Machine Learning is a subset of AI that uses algorithms to learn from data.", "topic: ML"),
        ("Deep Learning uses neural networks with many layers for complex pattern recognition tasks.", "topic: DL"),
        ("Natural Language Processing (NLP) enables computers to understand human language.", "topic: NLP"),
        ("Computer Vision deals with the interpretation and analysis of visual data.", "topic: CV"),
        ("Retrieval-Augmented Generation (RAG) combines information retrieval with text generation.", "topic: RAG"),
        ("LangChain is a framework for developing applications with language models.", "topic: LangChain"),
        ("DuckDB is an embedded analytical database with SQL support.", "topic: DuckDB"),
        ("Vector search enables semantic similarity searches in high-dimensional data spaces.", "topic: VectorSearch"),
        ("SmolLM3 is a compact but powerful open-source language model.", "topic: SmolLM3")
    ]
    return (sample_docs,)


@app.cell
def _(con, embedder, mo, np, sample_docs):
    # Embed and store documents
    for i, (content, metadata) in enumerate(sample_docs):
        embedding = embedder.encode([content])[0]
        con.execute("""
            INSERT OR REPLACE INTO documents (id, content, metadata, embedding) 
            VALUES (?, ?, ?, ?)
        """, (i, content, metadata, embedding.astype(np.float32)))

    doc_count = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    mo.md(f"📚 **{doc_count} documents with embeddings stored and indexed**")
    return


@app.cell
def _(con, embedder, mo, np):
    from langchain.tools import tool
    # Cell 8: LangChain Tools for RAG Agent
    @tool
    def search_documents(query: str) -> str:
        """Search the document collection for relevant information to a question."""
        # Create query embedding
        query_embedding = embedder.encode([query])[0]
    
        # Semantic search with DuckDB VSS
        results = con.execute("""
            SELECT content, metadata, array_cosine_distance(embedding, ?::FLOAT[384]) as distance
            FROM documents 
            ORDER BY distance ASC 
            LIMIT 3
        """, (query_embedding.astype(np.float32),)).fetchall()
    
        if not results:
            return "No relevant documents found."
    
        # Format results
        formatted_results = []
        for content, metadata, distance in results:
            formatted_results.append(f"📄 {content} ({metadata}, Similarity: {1-distance:.3f})")
    
        return "\n".join(formatted_results)

    @tool
    def get_document_statistics() -> str:
        """Returns statistics about the document collection."""
        stats = con.execute("""
            SELECT 
                COUNT(*) as total_docs,
                COUNT(DISTINCT metadata) as unique_topics
            FROM documents
        """).fetchone()
    
        return f"📊 Document collection: {stats[0]} documents, {stats[1]} different topics"

    @tool
    def add_document(content: str, metadata: str = "user_added") -> str:
        """Adds a new document to the collection."""
        # Find next available ID
        max_id = con.execute("SELECT COALESCE(MAX(id), -1) FROM documents").fetchone()[0]
        new_id = max_id + 1
    
        # Create embedding
        embedding = embedder.encode([content])[0]
    
        # Store document
        con.execute("""
            INSERT INTO documents (id, content, metadata, embedding) 
            VALUES (?, ?, ?, ?)
        """, (new_id, content, metadata, embedding.astype(np.float32)))
    
        return f"✅ Document successfully added (ID: {new_id})"

    tools = [search_documents, get_document_statistics, add_document]
    mo.md("🔧 **LangChain Tools for RAG Agent created**")

    return (tools,)


@app.cell
def _():
    from langchain.memory import ConversationBufferWindowMemory
    # Conversational Memory for the Agent
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,  # Keep the last 5 interactions
        return_messages=True
    )
    return (memory,)


@app.cell
def _():
    system_prompt = """You are a helpful RAG agent with access to a knowledge base.
    You are the Marimo Flow Agent, an AI RAG agent with access to a knowledge base, specifically designed to help users work with Marimo notebooks. Your task is to optimize the workflow by answering questions, making code suggestions, and providing advice on improving notebooks. You understand the principles of reactive programming in Marimo and use them to increase the efficiency and quality of notebook creation. Always remember that you are part of the Marimo ecosystem, especially regarding MLflow integrations.

    IMPORTANT RULES:
    1. ALWAYS use the search_documents tool before answering questions
    2. Cite the found sources in your answers
    3. If you don't find relevant information, say so honestly
    4. You can also add new documents if the user wants
    5. Answer precisely and helpfully in English

    Available Tools:
    - search_documents: Search the document collection
    - get_document_statistics: Statistics about the collection
    - add_document: Add new documents
    """
    return (system_prompt,)


@app.cell
def _(AgentType, initialize_agent, llm, memory, mo, tools):
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True
    )

    mo.md("🤖 **RAG Agent with Memory successfully configured**")
    return (agent,)


@app.cell
def _(con, np, query_emb):
    # 6. Similarity Search in DuckDB (Cosine similarity as example)
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


    results = []
    for row in con.execute("SELECT id, text, embedding FROM docs").fetchall():
        embs = np.frombuffer(row[2], dtype=np.float32)
        sim = cosine_similarity(query_emb, embs)
        results.append((sim, row[1]))
    return (results,)


@app.cell
def _(results):
    # 7. Sort and select the most similar texts
    results.sort(reverse=True)
    retrieved_text = results[0][1]  # Most similar
    return (retrieved_text,)


@app.cell
def _(model, query, retrieved_text, tokenizer):
    prompt = f"Context: {retrieved_text}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=256)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    return


@app.cell
def _():
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="ai/smollm3", base_url="http://localhost:12434/engines/v1", api_key="ignored"
    )
    return (llm,)


@app.cell
def _(agent, dataset_to_chart, find_docs, is_dataset, query, template):

    result = agent.invoke("How do I program a Plotly chart in Python?")
    print(result)


    def my_rag_model(messages, config):
        # Each message has a `content` attribute, as well as a `role`
        # attribute ("user", "system", "assistant");
        question = messages[-1].content
        docs = find_docs(question)
        prompt = template(question, docs, messages)
        response = query(prompt)
        if is_dataset(response):
            return dataset_to_chart(response)
        return response
    return


@app.cell
def _(mo, system_prompt):
    chat = mo.ui.chat(
        mo.ai.llm.openai(
            "ai/smollm3:latest",
            api_key="marimo",
            # Change this if you are using a different OpenAI-compatible endpoint.
            base_url="http://localhost:12434/engines/v1",
            system_message=system_prompt,
        ),
        prompts=["Write a haiku about recursion in programming."],
    )
    chat
    return


@app.cell
def _():
    return


@app.cell
def _():
    import altair as alt
    import polars as pl

    # Create a sample dataset
    data = {
        "Month": ["January", "February", "March", "April", "May", "June"],
        "Sunny Days": [10, 8, 12, 15, 11, 9],
    }

    # Convert to DataFrame
    df = pl.DataFrame(data)

    # Create the chart
    weather_chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="Month:N",  # Month on the x-axis (as a categorical variable)
            y="Sunny Days:Q",  # Number of sunny days on the y-axis (as a quantitative variable)
            tooltip=["Month", "Sunny Days"],  # Add hover information
            color="Month",  # Color by month
        )
    )

    # Display the chart
    weather_chart.display()
    return


if __name__ == "__main__":
    app.run()
