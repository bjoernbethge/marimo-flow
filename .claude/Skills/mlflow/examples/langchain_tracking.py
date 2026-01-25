"""
LangChain with MLflow Autologging Template

This example demonstrates how to use MLflow autologging with LangChain
for automatic tracking of LLM calls, chains, and agents.
"""

import mlflow
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import Tool
import os

# Configure MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("langchain-tracking-demo")

# Enable LangChain autologging
mlflow.langchain.autolog()


def simple_chain_example():
    """Basic LangChain chain with autologging"""

    # Set active model for linking traces
    mlflow.set_active_model(name="simple_qa_chain")

    # Build chain
    prompt = ChatPromptTemplate.from_template("Answer this question: {question}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    chain = prompt | model | StrOutputParser()

    # Use chain - automatically traced
    response = chain.invoke({"question": "What is MLflow?"})
    print(f"Response: {response}")

    # Log chain as model
    with mlflow.start_run():
        mlflow.langchain.log_model(
            lc_model=chain,
            name="qa_chain",
            params={"temperature": 0.7, "model": "gpt-4o-mini"},
            model_type="chain",
            registered_model_name="SimpleQAChain"
        )


def multi_step_chain_example():
    """Multi-step chain with intermediate tracking"""

    mlflow.set_active_model(name="multi_step_chain")

    # Create prompts for different steps
    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize this text: {text}"
    )
    analyze_prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of: {summary}"
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    # Build multi-step chain
    summarize_chain = summarize_prompt | model | StrOutputParser()
    analyze_chain = analyze_prompt | model | StrOutputParser()

    # Execute with automatic tracing
    text = "MLflow is an amazing platform for ML lifecycle management."

    with mlflow.start_run():
        summary = summarize_chain.invoke({"text": text})
        mlflow.log_param("input_length", len(text))
        mlflow.log_param("summary_length", len(summary))

        sentiment = analyze_chain.invoke({"summary": summary})

        mlflow.log_text(text, "input.txt")
        mlflow.log_text(summary, "summary.txt")
        mlflow.log_text(sentiment, "sentiment.txt")

        print(f"Summary: {summary}")
        print(f"Sentiment: {sentiment}")


def agent_with_tools_example():
    """LangChain agent with tools"""

    # Define custom tools
    def calculator(expression: str) -> str:
        """Calculate mathematical expression"""
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def get_weather(location: str) -> str:
        """Get weather for location (mock)"""
        return f"The weather in {location} is sunny and 72°F"

    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Calculate mathematical expressions"
        ),
        Tool(
            name="Weather",
            func=get_weather,
            description="Get weather for a location"
        )
    ]

    # Create agent
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to tools."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_openai_functions_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    mlflow.set_active_model(name="tool_agent")

    # Execute agent - all tool calls traced
    with mlflow.start_run():
        result = agent_executor.invoke({
            "input": "What is 25 * 4 and what's the weather in Paris?"
        })

        print(f"Result: {result['output']}")

        # Log agent as model
        mlflow.langchain.log_model(
            lc_model=agent_executor,
            name="tool_agent",
            params={"model": "gpt-4o", "tools": ["Calculator", "Weather"]},
            model_type="agent",
            registered_model_name="ToolAgent"
        )


def production_chain_example():
    """Production-ready chain with comprehensive tracking"""

    mlflow.set_active_model(name="production_qa")

    # Build production chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful customer support agent. Be concise and accurate."),
        ("human", "{question}")
    ])

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=500
    )

    chain = prompt | model | StrOutputParser()

    # Production usage with context
    def handle_customer_query(question: str, user_id: str, session_id: str):
        with mlflow.start_run():
            # Add context tags
            mlflow.set_tags({
                "user_id": user_id,
                "session_id": session_id,
                "environment": "production",
                "chain_type": "customer_support"
            })

            mlflow.update_current_trace(
                tags={
                    "mlflow.trace.user": user_id,
                    "mlflow.trace.session": session_id
                }
            )

            # Invoke chain
            response = chain.invoke({"question": question})

            # Log metrics
            mlflow.log_metric("response_length", len(response))
            mlflow.log_metric("question_length", len(question))

            # Get trace for token tracking
            trace_id = mlflow.get_last_active_trace_id()
            trace = mlflow.get_trace(trace_id=trace_id)

            if trace and hasattr(trace.info, 'token_usage'):
                tokens = trace.info.token_usage
                mlflow.log_metric("total_tokens", tokens['total_tokens'])
                mlflow.log_metric("input_tokens", tokens['input_tokens'])
                mlflow.log_metric("output_tokens", tokens['output_tokens'])

            return response

    # Example usage
    response = handle_customer_query(
        question="How do I reset my password?",
        user_id="user-123",
        session_id="session-abc"
    )
    print(f"Response: {response}")


if __name__ == "__main__":
    print("=" * 60)
    print("LangChain + MLflow Autologging Examples")
    print("=" * 60)

    print("\n1. Simple Chain Example")
    print("-" * 60)
    simple_chain_example()

    print("\n2. Multi-Step Chain Example")
    print("-" * 60)
    multi_step_chain_example()

    print("\n3. Agent with Tools Example")
    print("-" * 60)
    agent_with_tools_example()

    print("\n4. Production Chain Example")
    print("-" * 60)
    production_chain_example()

    print("\n" + "=" * 60)
    print("Check MLflow UI at http://localhost:5000")
    print("=" * 60)
