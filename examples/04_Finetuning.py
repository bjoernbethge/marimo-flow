import marimo

__generated_with = "0.14.10"
app = marimo.App(width="columns", sql_output="native")


@app.cell
def introduction():
    import marimo as mo
    
    mo.md("""
    # Fine-Tuning Open-Source LLM using QLoRA with MLflow and PEFT
    
    This notebook demonstrates how to fine-tune a large language model using **QLoRA** (Quantized LoRA) 
    and **PEFT** (Parameter Efficient Fine-Tuning) techniques, integrated with **MLflow** for experiment tracking.
    
    ## Overview
    - **Model**: Mistral-7B-v0.1 (7 billion parameters)
    - **Task**: Text-to-SQL generation
    - **Technique**: QLoRA for memory-efficient fine-tuning
    - **Dataset**: `b-mc2/sql-create-context` (78.6k SQL pairs)
    - **Hardware**: Single GPU with 20GB+ VRAM recommended
    
    ## Key Benefits
    - Train large models on limited hardware
    - Efficient parameter updates with LoRA
    - 4-bit quantization for memory savings
    - Complete experiment tracking with MLflow
    """)
    return (mo,)


@app.cell
def environment_setup(mo):
    mo.md("""
    ## 1. Environment Setup and Dependencies
    
    Install required libraries for fine-tuning:
    """)
    return


@app.cell
def check_gpu(mo):
    # Check GPU availability
    import subprocess
    import sys
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        gpu_info = result.stdout
        mo.md(f"""
        ### GPU Information
        ```
        {gpu_info}
        ```
        """)
    except FileNotFoundError:
        mo.md("⚠️ **Warning**: nvidia-smi not found. GPU may not be available.")
    return gpu_info, result, subprocess, sys


@app.cell
def dependencies_info(mo):
    mo.md("""
    ### Install Dependencies
    
    Run the following command to install required packages:
    
    ```bash
    pip install mlflow>=2.11.0 transformers peft accelerate bitsandbytes datasets torch numpy pandas
    ```
    
    Or if you're using this project's environment:
    
    ```bash
    pip install -e .
    ```
    """)
    return


@app.cell
def import_libraries(mo):
    # Import all required libraries
    try:
        import mlflow
        import numpy as np
        import pandas as pd
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        from datasets import load_dataset
        
        mo.md(f"""
        ### Library Versions
        - MLflow: {mlflow.__version__}
        - PyTorch: {torch.__version__}
        - CUDA Available: {torch.cuda.is_available()}
        - GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}
        
        ✅ All dependencies loaded successfully!
        """)
        
        return (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            LoraConfig,
            Trainer,
            TrainingArguments,
            get_peft_model,
            load_dataset,
            mlflow,
            np,
            pd,
            prepare_model_for_kbit_training,
            torch,
        )
        
    except ImportError as e:
        mo.md(f"""
        ❌ **Import Error**: {str(e)}
        
        Please install the required dependencies:
        
        ```bash
        pip install mlflow>=2.11.0 transformers peft accelerate bitsandbytes datasets torch numpy pandas
        ```
        
        Or update your environment to include these packages.
        """)
        
        # Return dummy objects to prevent further errors
        return tuple([None] * 14)


@app.cell
def dataset_preparation(mo):
    mo.md("""
    ## 2. Dataset Preparation
    
    Loading the SQL dataset from HuggingFace Hub:
    """)
    return


@app.cell
def load_dataset_data(load_dataset, pd, mo):
    # Load and explore the dataset
    dataset_name = "b-mc2/sql-create-context"
    dataset = load_dataset(dataset_name, split="train")
    
    # Display first few examples
    df_sample = pd.DataFrame(dataset.select(range(3)))
    
    mo.md(f"""
    ### Dataset: {dataset_name}
    
    **Size**: {len(dataset):,} samples
    
    **Columns**:
    - `question`: Natural language query
    - `context`: Database schema information  
    - `answer`: Expected SQL query
    
    **Sample Data**:
    """)
    return dataset, dataset_name, df_sample


@app.cell
def display_sample_data(dataset, mo):
    # Display sample data in a nice format
    sample_data = dataset.select(range(3))
    
    for i, example in enumerate(sample_data):
        mo.md(f"""
        **Example {i+1}:**
        
        **Question**: {example['question']}
        
        **Context**: 
        ```sql
        {example['context']}
        ```
        
        **Answer**: 
        ```sql
        {example['answer']}
        ```
        ---
        """)
        if i == 0:  # Only show first example to save space
            break
    return example, i, sample_data


@app.cell
def data_preprocessing_info(mo):
    mo.md("""
    ## 3. Data Preprocessing
    
    Create train/test split and define prompt template:
    """)
    return


@app.cell
def split_and_template(dataset, mo):
    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    # Define prompt template
    prompt_template = """You are a powerful text-to-SQL model. Given the SQL tables and natural language question, your job is to write SQL query that answers the question.

### Table:
{context}

### Question:
{question}

### Response:
{answer}"""

    mo.md(f"""
    ### Dataset Split
    - **Training samples**: {len(train_dataset):,}
    - **Evaluation samples**: {len(eval_dataset):,}
    
    ### Prompt Template
    ```
    {prompt_template}
    ```
    """)
    return eval_dataset, prompt_template, train_dataset, train_test_split


@app.cell
def preprocess_data(train_dataset, eval_dataset, prompt_template, mo):
    # Preprocess data function
    def preprocess_function(examples):
        inputs = []
        for question, context, answer in zip(examples['question'], examples['context'], examples['answer']):
            prompt = prompt_template.format(
                context=context,
                question=question, 
                answer=answer
            )
            inputs.append(prompt)
        return {"text": inputs}
    
    # Apply preprocessing
    train_dataset_processed = train_dataset.map(preprocess_function, batched=True)
    eval_dataset_processed = eval_dataset.map(preprocess_function, batched=True)
    
    mo.md("""
    ✅ **Data preprocessing completed**
    
    Each sample now contains the formatted prompt with question, context, and expected SQL answer.
    """)
    return preprocess_function, train_dataset_processed, eval_dataset_processed


@app.cell
def model_setup_info(mo):
    mo.md("""
    ## 4. Model and Tokenizer Setup
    
    Load Mistral-7B with 4-bit quantization for efficient training:
    """)
    return


@app.cell
def configure_model(BitsAndBytesConfig, AutoTokenizer, torch, mo):
    # Model configuration
    model_name = "mistralai/Mistral-7B-v0.1"
    
    # 4-bit quantization config for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    mo.md(f"""
    ### Model Configuration
    - **Base Model**: {model_name}
    - **Quantization**: 4-bit NF4 with double quantization
    - **Compute dtype**: bfloat16
    - **Tokenizer**: LlamaTokenizerFast
    
    ⚠️ **Note**: Loading the model will require significant GPU memory (10-15GB)
    """)
    return bnb_config, model_name, tokenizer


@app.cell
def load_model(AutoModelForCausalLM, model_name, bnb_config, prepare_model_for_kbit_training, mo):
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    mo.md(f"""
    ✅ **Model loaded successfully**
    
    - Model device: {next(model.parameters()).device}
    - Model dtype: {next(model.parameters()).dtype}
    - Trainable parameters will be added via LoRA
    """)
    return (model,)


@app.cell
def lora_config_info(mo):
    mo.md("""
    ## 5. LoRA Configuration
    
    Configure Low-Rank Adaptation for efficient fine-tuning:
    """)
    return


@app.cell
def setup_lora(model, LoraConfig, get_peft_model, mo):
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,                    # Rank of adaptation
        lora_alpha=32,           # LoRA scaling parameter
        target_modules=[         # Modules to apply LoRA to
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        lora_dropout=0.1,        # LoRA dropout
        bias="none",             # Bias type
        task_type="CAUSAL_LM",   # Task type
    )
    
    # Apply LoRA to model
    model_with_lora = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_with_lora.parameters())
    
    mo.md(f"""
    ### LoRA Configuration
    - **Rank (r)**: {lora_config.r}
    - **Alpha**: {lora_config.lora_alpha}
    - **Dropout**: {lora_config.lora_dropout}
    - **Target modules**: {len(lora_config.target_modules)} modules
    
    ### Parameter Efficiency
    - **Trainable parameters**: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)
    - **Total parameters**: {total_params:,}
    - **Memory reduction**: ~{(1-trainable_params/total_params)*100:.1f}%
    """)
    return lora_config, total_params, trainable_params, model_with_lora


@app.cell
def tokenization_info(mo):
    mo.md("""
    ## 6. Data Tokenization
    
    Tokenize the training data:
    """)
    return


@app.cell
def tokenize_data(train_dataset_processed, eval_dataset_processed, tokenizer, DataCollatorForLanguageModeling, mo):
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=512,
            return_overflowing_tokens=False,
        )
    
    # Tokenize datasets
    tokenized_train = train_dataset_processed.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset_processed.map(tokenize_function, batched=True)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
    )
    
    mo.md(f"""
    ✅ **Tokenization completed**
    
    - **Max sequence length**: 512 tokens
    - **Training samples**: {len(tokenized_train):,}
    - **Evaluation samples**: {len(tokenized_eval):,}
    - **Data collator**: Language modeling (causal)
    """)
    return data_collator, tokenize_function, tokenized_eval, tokenized_train


@app.cell
def training_config_info(mo):
    mo.md("""
    ## 7. Training Configuration
    
    Set up training arguments and MLflow experiment:
    """)
    return


@app.cell
def setup_training(mlflow, TrainingArguments, mo):
    # MLflow experiment setup
    experiment_name = "LLM_Fine_Tuning_QLoRA"
    mlflow.set_experiment(experiment_name)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,              # Start with 1 epoch for demo
        per_device_train_batch_size=4,   # Adjust based on GPU memory
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,   # Effective batch size = 4*4=16
        learning_rate=2e-4,
        weight_decay=0.001,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=500,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",       # Memory efficient optimizer
        remove_unused_columns=False,
        report_to="mlflow",             # Log to MLflow
        run_name="Mistral-7B-SQL-QLoRA",
    )
    
    mo.md(f"""
    ### Training Configuration
    - **Epochs**: {training_args.num_train_epochs}
    - **Batch size**: {training_args.per_device_train_batch_size} (per device)
    - **Gradient accumulation**: {training_args.gradient_accumulation_steps} steps
    - **Effective batch size**: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}
    - **Learning rate**: {training_args.learning_rate}
    - **Optimizer**: {training_args.optim}
    - **MLflow experiment**: {experiment_name}
    """)
    return experiment_name, training_args


@app.cell
def trainer_setup_info(mo):
    mo.md("""
    ## 8. Training Setup
    
    Initialize the trainer:
    """)
    return


@app.cell
def initialize_trainer(model_with_lora, training_args, tokenized_train, tokenized_eval, tokenizer, data_collator, Trainer, mo):
    # Initialize trainer
    trainer = Trainer(
        model=model_with_lora,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    mo.md("""
    ✅ **Trainer initialized**
    
    Ready to start training! The trainer includes:
    - Model with LoRA adapters
    - Training and evaluation datasets
    - MLflow integration for experiment tracking
    """)
    return (trainer,)


@app.cell
def training_controls(mo):
    # Training controls
    start_training = mo.ui.button(
        label="🚀 Start Training",
        kind="success",
        disabled=False
    )
    
    stop_training = mo.ui.button(
        label="⏹️ Stop Training", 
        kind="danger",
        disabled=True
    )
    
    mo.md(f"""
    ## 9. Training Execution
    
    **Warning**: Training will take significant time and GPU resources.
    
    {mo.hstack([start_training, stop_training])}
    
    Monitor training progress in the MLflow UI:
    ```bash
    mlflow ui
    ```
    Then visit http://localhost:5000
    """)
    return start_training, stop_training


@app.cell
def execute_training(start_training, trainer, mlflow, model_name, lora_config, dataset_name, tokenized_train, tokenized_eval, mo):
    # Training execution logic
    training_results = None
    
    if start_training.value:
        mo.md("""
        🔄 **Training in progress...**
        
        This may take 30+ minutes depending on your hardware.
        Check MLflow UI for real-time metrics.
        """)
        
        try:
            # Start MLflow run
            with mlflow.start_run():
                # Train the model
                training_results = trainer.train()
                
                # Log additional metrics
                mlflow.log_params({
                    "model_name": model_name,
                    "lora_r": lora_config.r,
                    "lora_alpha": lora_config.lora_alpha,
                    "dataset": dataset_name,
                    "train_samples": len(tokenized_train),
                    "eval_samples": len(tokenized_eval),
                })
                
        except Exception as e:
            mo.md(f"""
            ❌ **Training failed**: {str(e)}
            
            This might be due to insufficient GPU memory or other issues.
            """)
    else:
        mo.md("""
        ⏸️ **Training not started**
        
        Click the "Start Training" button above to begin fine-tuning.
        """)
    return (training_results,)


@app.cell
def model_saving_info(mo):
    mo.md("""
    ## 10. Model Saving and Evaluation
    
    Save the fine-tuned model to MLflow:
    """)
    return


@app.cell
def save_model_controls(mo):
    save_model = mo.ui.button(
        label="💾 Save Model to MLflow",
        kind="success"
    )
    
    mo.hstack([save_model])
    return (save_model,)


@app.cell
def save_model_logic(save_model, training_results, mlflow, trainer, tokenizer, mo):
    # Save model logic
    if save_model.value and training_results is not None:
        try:
            # Define input example for model signature
            input_example = "### Table:\nCREATE TABLE users (id INT, name VARCHAR)\n### Question:\nHow many users are there?\n### Response:"
            
            # Save the PEFT model
            mlflow.transformers.log_model(
                transformers_model={
                    "model": trainer.model,
                    "tokenizer": tokenizer,
                },
                artifact_path="model",
                input_example=input_example,
                inference_config={
                    "max_new_tokens": 256,
                    "repetition_penalty": 1.15,
                    "return_full_text": False,
                }
            )
            
            mo.md("""
            ✅ **Model saved successfully to MLflow!**
            
            The model includes:
            - LoRA adapters
            - Tokenizer
            - Inference configuration
            - Input/output signature
            """)
            
        except Exception as e:
            mo.md(f"❌ **Error saving model**: {str(e)}")
    
    elif save_model.value:
        mo.md("⚠️ **No trained model to save**. Please complete training first.")
    return input_example,


@app.cell
def model_testing_info(mo):
    mo.md("""
    ## 11. Model Testing
    
    Test the fine-tuned model on sample queries:
    """)
    return


@app.cell
def test_model_controls(mo):
    # Test query input
    test_question = mo.ui.text_area(
        placeholder="Enter a natural language question about SQL...",
        label="Test Question",
        value="When Essendon played away; where did they play?"
    )
    
    test_context = mo.ui.text_area(
        placeholder="Enter table schema...",
        label="Table Context", 
        value="CREATE TABLE table_name_50 (venue VARCHAR, away_team VARCHAR)"
    )
    
    run_test = mo.ui.button(label="🧪 Test Model", kind="success")
    
    mo.md(f"""
    ### Test the Fine-tuned Model
    
    {test_context}
    
    {test_question}
    
    {run_test}
    """)
    return run_test, test_context, test_question


@app.cell
def test_model_logic(run_test, training_results, test_context, test_question, tokenizer, model_with_lora, torch, mo):
    # Model testing logic
    if run_test.value and training_results is not None:
        try:
            # Format test prompt
            test_prompt = f"""You are a powerful text-to-SQL model. Given the SQL tables and natural language question, your job is to write SQL query that answers the question.

### Table:
{test_context.value}

### Question:
{test_question.value}

### Response:
"""
            
            # Tokenize input
            inputs = tokenizer(test_prompt, return_tensors="pt").to(model_with_lora.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model_with_lora.generate(
                    **inputs,
                    max_new_tokens=256,
                    repetition_penalty=1.15,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_sql = response[len(test_prompt):].strip()
            
            mo.md(f"""
            ### Generated SQL Query
            
            ```sql
            {generated_sql}
            ```
            
            **Full Response**:
            ```
            {response}
            ```
            """)
            
        except Exception as e:
            mo.md(f"❌ **Error during inference**: {str(e)}")
    
    elif run_test.value:
        mo.md("⚠️ **No trained model available**. Please complete training first.")
    return generated_sql, inputs, outputs, response, test_prompt


@app.cell
def results_and_next_steps(mo):
    mo.md("""
    ## 12. Results and Next Steps
    
    ### What You've Accomplished
    
    ✅ **Efficient Fine-tuning**: Used QLoRA to fine-tune a 7B parameter model with minimal GPU memory
    
    ✅ **Parameter Efficiency**: Only trained ~1% of total parameters using LoRA adapters
    
    ✅ **Experiment Tracking**: Complete MLflow integration with metrics, parameters, and model artifacts
    
    ✅ **Production Ready**: Model saved with inference configuration and signatures
    
    ### Performance Optimization
    
    The QLoRA technique provides several advantages:
    - **Memory Efficiency**: 4-bit quantization reduces memory usage by ~75%
    - **Training Speed**: LoRA adapters train much faster than full fine-tuning
    - **Quality**: Achieves comparable performance to full fine-tuning
    - **Flexibility**: Easy to swap adapters for different tasks
    
    ### Next Steps
    
    1. **Evaluate Model**: Use MLflow's evaluation features to assess performance
    2. **Deploy Model**: Use MLflow model serving for production deployment
    3. **Experiment**: Try different LoRA configurations, datasets, or base models
    4. **Scale**: Use multiple GPUs or larger models for better performance
    
    ### MLflow Integration Benefits
    
    - **Reproducibility**: All hyperparameters and configurations logged
    - **Comparison**: Easy to compare different training runs
    - **Deployment**: Streamlined model deployment pipeline
    - **Collaboration**: Share experiments and models with team members
    """)
    return


@app.cell
def additional_resources(mo):
    mo.md("""
    ## 13. Additional Resources
    
    ### Documentation Links
    - [MLflow Transformers Documentation](https://mlflow.org/docs/latest/models.html#transformers-transformers)
    - [PEFT Library Documentation](https://huggingface.co/docs/peft/index)
    - [QLoRA Paper](https://arxiv.org/abs/2305.14314)
    - [Mistral-7B Model](https://huggingface.co/mistralai/Mistral-7B-v0.1)
    
    ### Marimo Features Used
    - Interactive UI elements for training control
    - Real-time progress monitoring
    - Integrated code execution and documentation
    - Dynamic content based on training state
    
    ### Optimization Tips
    1. **Memory**: Reduce batch size if OOM errors occur
    2. **Speed**: Increase batch size and gradient accumulation for faster training
    3. **Quality**: Train for more epochs with learning rate scheduling
    4. **Efficiency**: Experiment with different LoRA ranks and targets
    """)
    return


if __name__ == "__main__":
    app.run()
