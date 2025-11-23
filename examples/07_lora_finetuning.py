# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "bitsandbytes>=0.44.0",
#     "datasets>=3.0.0",
#     "huggingface-hub[cli,hf-transfer,hf-xet,mcp,torch]>=0.26.0",
#     "marimo",
#     "mlflow[genai,langchain]>=2.17.0",
#     "peft>=0.13.0",
#     "polars[database,pyarrow,sqlalchemy]>=1.12.0",
#     "torch[opt-einsum,optree]>=2.5.0",
#     "transformers[accelerate,onnxruntime,optuna,torch]>=4.46.0",
# ]
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    # Fine-Tuning Modern SLMs

    A reactive notebook for fine-tuning state-of-the-art **Small Language Models (2025)**:
    - **SmolLM3-3B**: Latest 3B model with 64k context, tool calling, reasoning mode
    - **Phi-4**: Microsoft's compact reasoning powerhouse  
    - **Mistral-Small-24B**: State-of-the-art efficiency for the size

    Using modern techniques:
    - **QLoRA**: Memory-efficient 4-bit quantization
    - **PEFT**: Parameter-efficient fine-tuning
    - **MLflow**: Experiment tracking
    """
    )
    return


@app.cell
def _(mo):
    # Configuration controls with text inputs for URLs
    model_input = mo.ui.text(
        value="HuggingFaceTB/SmolLM3-3B",
        label="Model URL/Name",
        placeholder="e.g. HuggingFaceTB/SmolLM3-3B, microsoft/Phi-4, mistralai/Mistral-Small-24B-Instruct-2501"
    )

    dataset_input = mo.ui.text(
        value="HuggingFaceH4/no_robots", 
        label="Dataset URL/Name",
        placeholder="e.g. HuggingFaceH4/no_robots, microsoft/orca-math-word-problems-200k, argilla/synth-apigen-v0.1"
    )

    epochs_slider = mo.ui.slider(
        start=1, stop=5, value=1, step=1,
        label="Training Epochs"
    )

    batch_size_slider = mo.ui.slider(
        start=1, stop=8, value=4, step=1,
        label="Batch Size"
    )

    mo.output.append(mo.md(f"""
    ## Configuration

        {model_input}
        {dataset_input}
        {epochs_slider}
        {batch_size_slider}

    💡 **Tip**: Try modern 2025 models like SmolLM3-3B (tool calling, reasoning) or Phi-4 (math reasoning)
    """))
    return batch_size_slider, dataset_input, epochs_slider, model_input


@app.cell
def _(mo):
    # Setup HuggingFace environment variables for better control
    import os

    # Set HuggingFace environment variables based on official docs
    # https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
    project_models_dir = os.path.join(os.getcwd(), "models")

    # Modern HF environment variables
    os.environ['HF_HOME'] = project_models_dir  # Main HF cache directory
    os.environ['HF_HUB_CACHE'] = os.path.join(project_models_dir, "hub")  # Models/datasets cache
    os.environ['HF_ASSETS_CACHE'] = os.path.join(project_models_dir, "assets")  # Preprocessed data
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(project_models_dir, "sentence-transformers")

    # Performance optimizations
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # Faster downloads with hf_transfer
    os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'  # High performance for hf_xet

    # Optional: Set token from environment if available
    if os.environ.get('HF_TOKEN'):
        mo.output.append(mo.md("🔑 **HF_TOKEN found** in environment variables"))

    # Create directories if they don't exist
    for cache_dir in [project_models_dir, os.environ['HF_HUB_CACHE'], os.environ['HF_ASSETS_CACHE']]:
        os.makedirs(cache_dir, exist_ok=True)

    # Installation check
    try:
        import datasets
        import mlflow
        import peft
        import polars as pl
        import torch
        import transformers
        from huggingface_hub import HfFolder

        # Check if HuggingFace token is available
        try:
            token = HfFolder.get_token()
            hf_status = "✅ Authenticated" if token else "⚠️ No Token"
        except:
            hf_status = "⚠️ No Token"

        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()

        status = "✅ Ready" if gpu_available else "⚠️ CPU Only"

        mo.output.append(mo.md(f"""
        ## Environment Status: {status}

        - **PyTorch**: {torch.__version__}
        - **Transformers**: {transformers.__version__}
        - **MLflow**: {mlflow.__version__}
        - **HuggingFace**: {hf_status}
        - **Models Directory**: `{project_models_dir}`
        - **GPU Available**: {gpu_available}
        - **GPU Count**: {gpu_count}

        💡 **Tip**: Set HF_TOKEN environment variable or use `huggingface-cli login` for model access
        """))

    except ImportError as e:
        mo.md(f"""
        ## ❌ Missing Dependencies

        Please install required packages:
        ```bash
        uv add torch transformers datasets peft mlflow accelerate bitsandbytes huggingface-hub
        ```

        Error: {e}
        """)
        torch = transformers = datasets = peft = mlflow = None

    return datasets, mlflow, os, peft, pl, torch, transformers


@app.cell
def _(mo):
    hf_token_input = mo.ui.text(kind="password")
    return (hf_token_input,)


@app.cell
def _(hf_token_input, mo):
    from huggingface_hub import login ,list_datasets, list_models
    def on_click(value):
        login(token=hf_token_input.value)
    # HuggingFace Login Helper
    hf_login_btn = mo.ui.button(label="🔑 Login", kind="success", on_click=on_click)

    mo.md(f"""
    ## HuggingFace Authentication
    {hf_token_input}
    {hf_login_btn}

    💡 Create your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
    """)
    return (hf_login_btn,)


@app.cell
def _(dataset_input, datasets, hf_login_btn, mo, os, pl):

    if datasets is None:
        dataset = None
        mo.output.append(mo.md("❌ Cannot load dataset - missing dependencies"))
    else:
        # Handle login button click
        if hf_login_btn.value:
            try:
                from huggingface_hub import login as hf_login
                # This will prompt for token in the notebook
                hf_login()
                mo.output.append(mo.md("✅ **Login initiated** - Enter your token in the prompt above"))
            except Exception as e:
                mo.output.append(mo.md(f"❌ **Login error**: {str(e)}"))
        # Load dataset reactively based on input
        dataset_name = dataset_input.value

        try:
            # Check if it's a HuggingFace hf:// URL for CSV
            if dataset_name.startswith("hf://"):
                # Use polars for CSV loading from HuggingFace URLs
                dataset = pl.read_csv(dataset_name)

                mo.output.append(mo.md(f"""
                ## Dataset (Polars): {dataset_name}

                **Shape**: {dataset.shape}
                **Columns**: {list(dataset.columns)}

                ### Sample Data
                """))
                mo.output.append(mo.ui.dataframe(dataset.head(3)))

            else:
                # Regular HuggingFace dataset with local cache
                dataset = datasets.load_dataset(
                    dataset_name, 
                    split="train",
                    cache_dir=os.environ.get('HF_HUB_CACHE')
                )

                mo.output.append(mo.md(f"""
                ## Dataset: {dataset_name}

                **Size**: {len(dataset):,} samples
                **Columns**: {list(dataset.features.keys())}
                """))

        except Exception as e:
            mo.output.append(mo.md(f"❌ **Error loading dataset**: {str(e)}"))
            dataset = None

    return (dataset,)


@app.cell
def _(dataset, mo):
    if dataset is not None:
        # Show sample data using marimo's built-in dataframe viewer for regular datasets
        if hasattr(dataset, 'to_pandas'):  # HuggingFace dataset
            sample_df = dataset.to_pandas().head(3)
            mo.output.append(mo.md("### Sample Data (HuggingFace Dataset)"))
            mo.output.append(mo.ui.dataframe(sample_df))
        # Polars datasets are already handled in the previous cell
    else:
        mo.output.append(mo.md("No dataset loaded"))
    return


@app.cell
def _(dataset, mo, model_input, os, transformers):

    if dataset is None or transformers is None:
        tokenizer = None
        mo.output.append(mo.md("❌ Cannot load model - missing dependencies"))
    else:
        model_name = model_input.value

        try:
            # Load tokenizer with local directory support
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=os.environ.get('HF_HUB_CACHE'),
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            mo.output.append(mo.md(f"""
            ## Model Configuration

            **Model**: {model_name}
            **Tokenizer**: {type(tokenizer).__name__}
            **Vocab Size**: {tokenizer.vocab_size:,}
            **Cache Dir**: `{os.environ.get('HF_HUB_CACHE', 'default')}`
            """))
        except Exception as e:
            mo.output.append(mo.md(f"❌ **Error loading model**: {str(e)}"))
            model_name = None

    return (model_name,)


@app.cell
def _(mo, torch):
    if torch is not None:
        # Quantization configuration
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        mo.output.append(mo.md("""
        ## Quantization Config (QLoRA)

        - **4-bit quantization**: NF4 format
        - **Compute dtype**: bfloat16  
        - **Double quantization**: Enabled
        - **Memory savings**: ~75%
        """))
    else:
        bnb_config = None

    return (bnb_config,)


@app.cell(hide_code=True)
def _(mo, peft):
    if peft is not None:
        # LoRA configuration optimized for modern SLMs (2025)
        lora_config = peft.LoraConfig(
            r=32,  # Higher rank for better performance
            lora_alpha=64,  # Scaled alpha for stability
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,  # Lower dropout for small models
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=True,  # Rank-stabilized LoRA for better training
        )

        mo.output.append(mo.md(f"""
        ## LoRA Configuration

        - **Rank**: {lora_config.r}
        - **Alpha**: {lora_config.lora_alpha}  
        - **Dropout**: {lora_config.lora_dropout}
        - **Target modules**: {len(lora_config.target_modules)}
        """))
    else:
        lora_config = None

    return (lora_config,)


@app.cell
def _(mo):
    # Training controls
    start_btn = mo.ui.run_button(label="🚀 Start Training")
    stop_btn = mo.ui.button(label="⏹️ Stop", kind="danger")

    mo.md(f"""
    ## Training Controls

    {start_btn} {stop_btn}

    💡 **Tip**: Use the run button to prevent accidental expensive training runs
    """)
    return (start_btn,)


@app.cell
def _(
    batch_size_slider,
    bnb_config,
    dataset_input,
    epochs_slider,
    lora_config,
    mlflow,
    mo,
    model_name,
    os,
    peft,
    start_btn,
    transformers,
):

    if not start_btn.value:
        mo.output.append(mo.md("⏸️ Training not started. Click the button above to begin."))
        training_results = None
    else:
        if any(x is None for x in [transformers, mlflow, bnb_config, lora_config, peft]):
            mo.output.append(mo.md("❌ Cannot start training - missing dependencies"))
            training_results = None
        else:
            try:
                mo.output.append(mo.md("🔄 **Loading model and starting training...**"))

                # Load model with quantization and local cache
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    cache_dir=os.environ.get('HF_HUB_CACHE'),
                )

                # Prepare for training
                model = peft.prepare_model_for_kbit_training(model)
                model = peft.get_peft_model(model, lora_config)

                # Training arguments
                training_args = transformers.TrainingArguments(
                    output_dir="./results",
                    num_train_epochs=epochs_slider.value,
                    per_device_train_batch_size=batch_size_slider.value,
                    learning_rate=2e-4,
                    logging_steps=10,
                    report_to="mlflow",
                    run_name="marimo-finetuning",
                )

                # Real scenario: Prepare training data
                from transformers import DataCollatorForLanguageModeling

                # Get tokenizer for training
                training_tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=os.environ.get('HF_HUB_CACHE'),
                    trust_remote_code=True
                )
                if training_tokenizer.pad_token is None:
                    training_tokenizer.pad_token = training_tokenizer.eos_token

                # Prepare training dataset (simplified example)
                def tokenize_function(examples):
                    # For instruction datasets, combine instruction + response
                    if 'messages' in examples:
                        # Chat format
                        texts = [training_tokenizer.apply_chat_template(msg, tokenize=False) for msg in examples['messages']]
                    elif 'instruction' in examples and 'response' in examples:
                        # Instruction format
                        texts = [f"### Instruction:\n{inst}\n\n### Response:\n{resp}" 
                                for inst, resp in zip(examples['instruction'], examples['response'])]
                    else:
                        # Fallback: use text field or first string field
                        text_field = 'text' if 'text' in examples else next(iter(examples.keys()))
                        texts = examples[text_field]

                    return training_tokenizer(texts, truncation=True, padding=True, max_length=512)

                # Process dataset
                from datasets import load_dataset
                train_dataset = load_dataset(
                    dataset_input.value, 
                    split="train[:1000]",  # Limit for demo
                    cache_dir=os.environ.get('HF_HUB_CACHE')
                )

                # Tokenize dataset
                train_dataset = train_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=train_dataset.column_names
                )

                # Data collator
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=training_tokenizer,
                    mlm=False  # Causal LM, not masked
                )

                # Initialize trainer with real training setup
                trainer = transformers.Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    data_collator=data_collator,
                    tokenizer=training_tokenizer,
                )

                # Start MLflow run
                mlflow.set_experiment("Marimo-Fine-Tuning")

                with mlflow.start_run(run_name=f"SmolLM3-{dataset_input.value.split('/')[-1]}"):
                    # Log hyperparameters
                    mlflow.log_params({
                        "model_name": model_name,
                        "dataset": dataset_input.value,
                        "epochs": epochs_slider.value,
                        "batch_size": batch_size_slider.value,
                        "learning_rate": training_args.learning_rate,
                        "lora_rank": lora_config.r,
                        "lora_alpha": lora_config.lora_alpha,
                    })

                    mo.output.append(mo.md("🚀 **Starting real training...**"))

                    # Start actual training
                    trainer.train()

                    # Save the trained model
                    output_dir = f"./fine_tuned_{model_name.split('/')[-1]}"
                    trainer.save_model(output_dir)

                    # Log model to MLflow
                    mlflow.transformers.log_model(
                        transformers_model=trainer.model,
                        artifact_path="model",
                        task="text-generation"
                    )

                mo.output.append(mo.md(f"""
                ✅ **Training completed successfully!**

                - **Model**: {model_name}
                - **Dataset**: {dataset_input.value}
                - **Epochs**: {epochs_slider.value}
                - **Batch size**: {batch_size_slider.value}
                - **Trainable params**: {model.get_nb_trainable_parameters():,}
                - **Output dir**: `{output_dir}`
                - **MLflow run**: Active

                🎯 **Model saved and logged to MLflow!**
                """))

                training_results = {"status": "completed", "model_path": output_dir}

            except Exception as e:
                mo.output.append(mo.md(f"❌ **Training failed**: {str(e)}"))
                training_results = None

    return (training_results,)


@app.cell
def _(mo, training_results):
    if training_results is not None:
        # Model testing interface
        test_input = mo.ui.text_area(
            placeholder="Enter your SQL question here...",
            label="Test Question",
            value="How many users are in the database?"
        )

        test_btn = mo.ui.button(label="🧪 Test Model")

        mo.output.append(mo.md(f"""
        ## Model Testing

        {test_input}

        {test_btn}
        """))
    else:
        test_input = test_btn = None
        mo.md("Complete training first to test the model.")

    return test_btn, test_input


@app.cell
def _(mo, test_btn, test_input, training_results):
    if test_btn is not None and test_btn.value and training_results is not None:
        # Simulate model inference
        user_question = test_input.value
        generated_sql = f"SELECT COUNT(*) FROM users; -- Generated for: '{user_question}'"

        mo.output.append(mo.md(f"""
        ### Generated SQL

        **Question**: {user_question}

        **Generated SQL**:
        ```sql
        {generated_sql}
        ```

        *Note: This is a demo output. Real inference would use the trained model.*
        """))
    else:
        mo.md("Enter a question and click test to see results.")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Summary

    This reactive notebook demonstrates:

    ✅ **Proper marimo patterns**:
    - Reactive cells with correct dependencies
    - UI elements that trigger updates
    - `mo.ui.run_button()` for expensive operations
    - Built-in dataframe viewer

    ✅ **QLoRA fine-tuning setup**:
    - 4-bit quantization configuration
    - LoRA parameter-efficient training
    - MLflow experiment tracking

    ✅ **Interactive controls**:
    - Model and dataset selection
    - Hyperparameter sliders
    - Training start/stop buttons
    - Model testing interface

    ### Next Steps
    - Add actual dataset preprocessing
    - Implement full training loop
    - Add model evaluation metrics
    - Export trained model artifacts
    """
    )
    return


if __name__ == "__main__":
    app.run()
