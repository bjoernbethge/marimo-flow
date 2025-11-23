# /// script
# dependencies = [
#     "marimo",
#     "openvino>=2025.3.0",
#     "numpy>=1.26.4",
#     "pillow>=11.0.0",
#     "httpx>=0.27.0",
#     "pydantic>=2.11.7",
# ]
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import json
    import os
    from typing import Any, Dict, List, Optional

    import httpx
    import marimo as mo
    import numpy as np
    import openvino as ov
    from PIL import Image
    from pydantic import BaseModel
    return Image, httpx, io, json, mo, np, os, ov


@app.cell
def _(mo, ov):
    # Initialize OpenVINO runtime
    core = ov.Core()
    # Get available devices
    available_devices = core.available_devices
    get_devices, set_devices = mo.state(available_devices)
    get_selected_device, set_selected_device = mo.state(available_devices[0] if available_devices else "CPU")

    return core, get_devices, get_selected_device, set_selected_device


@app.cell
def _(mo):
    # Model management state
    get_model_list, set_model_list = mo.state([])
    get_selected_model, set_selected_model = mo.state("")
    get_model_info, set_model_info = mo.state({})

    return (
        get_model_list,
        get_selected_model,
        set_model_info,
        set_model_list,
        set_selected_model,
    )


@app.cell
def _(get_devices, get_selected_device, mo, set_selected_device):
    # Device selection UI
    device_dropdown = mo.ui.dropdown(
        options=get_devices(),
        label="🖥️ OpenVINO Device",
        value=get_selected_device(),
        on_change=set_selected_device
    )

    device_info = mo.md(f"""
    **Selected Device:** {get_selected_device()}

    **Available Devices:** {', '.join(get_devices())}
    """)

    return device_dropdown, device_info


@app.cell
def _(mo):
    # Model URL input
    model_url = mo.ui.text(
        label="🔗 Model URL", 
        full_width=True, 
        placeholder="https://huggingface.co/openvino/bert-base-uncased/resolve/main/model.xml"
    )

    # Model name input
    model_name = mo.ui.text(
        label="📝 Model Name",
        full_width=True,
        placeholder="bert-base-uncased"
    )

    return model_name, model_url


@app.cell
def _(
    core,
    get_model_list,
    get_selected_device,
    httpx,
    mo,
    model_name,
    model_url,
    os,
    set_model_list,
    set_selected_model,
):
    def download_and_load_model():
        if not model_url.value or not model_name.value:
            return

        try:
            # Download model if it's a URL
            if model_url.value.startswith('http'):
                model_path = f"models/{model_name.value}"
                os.makedirs("models", exist_ok=True)

                # Download XML and bin files
                xml_url = model_url.value
                bin_url = model_url.value.replace('.xml', '.bin')

                xml_response = httpx.get(xml_url)
                bin_response = httpx.get(bin_url)

                with open(f"{model_path}.xml", 'wb') as f:
                    f.write(xml_response.content)
                with open(f"{model_path}.bin", 'wb') as f:
                    f.write(bin_response.content)

                model_path = f"{model_path}.xml"
            else:
                model_path = model_url.value

            # Load model
            model = core.read_model(model_path)
            compiled_model = core.compile_model(model, get_selected_device())

            # Update model list
            current_models = get_model_list()
            if model_name.value not in current_models:
                current_models.append(model_name.value)
                set_model_list(current_models)
                set_selected_model(model_name.value)

            return f"✅ Model {model_name.value} loaded successfully on {get_selected_device()}"

        except Exception as e:
            return f"❌ Error loading model: {str(e)}"

    load_button = mo.ui.button(
        label="⬇️ Load Model", 
        value=model_url.value, 
        kind="warn", 
        on_change=download_and_load_model
    )

    load_ui = mo.vstack([model_name, model_url, load_button], justify="start")
    return (load_ui,)


@app.cell
def _(get_model_list, get_selected_model, mo, os, set_selected_model):
    def delete_model(model_name):
        try:
            # Remove model files
            model_path = f"models/{model_name}"
            if os.path.exists(f"{model_path}.xml"):
                os.remove(f"{model_path}.xml")
            if os.path.exists(f"{model_path}.bin"):
                os.remove(f"{model_path}.bin")

            return f"✅ Model {model_name} deleted successfully"
        except Exception as e:
            return f"❌ Error deleting model: {str(e)}"

    delete_button = mo.ui.button(
        label='❌ Delete Model', 
        value=get_selected_model(), 
        kind="danger", 
        on_change=delete_model
    )

    model_dropdown = mo.ui.dropdown(
        options=get_model_list(),
        label="🤖 Loaded Models",
        on_change=set_selected_model
    )

    return delete_button, model_dropdown


@app.cell
def _(core, get_selected_device, get_selected_model, mo, os, set_model_info):
    def show_model_info():
        if not get_selected_model():
            return mo.md("No model selected")

        try:
            model_path = f"models/{get_selected_model()}.xml"
            if not os.path.exists(model_path):
                return mo.md("Model file not found")

            model = core.read_model(model_path)
            compiled_model = core.compile_model(model, get_selected_device())

            # Get model information
            info = {
                "model_name": get_selected_model(),
                "device": get_selected_device(),
                "input_shape": str(model.inputs[0].shape),
                "output_shape": str(model.outputs[0].shape),
                "input_names": [input.get_any_name() for input in model.inputs],
                "output_names": [output.get_any_name() for output in model.outputs],
                "model_size": f"{os.path.getsize(model_path) / 1024:.2f} KB"
            }

            set_model_info(info)

            return mo.md(f"""
            **Model Information:**

            🧠 **Model:** {info['model_name']}
            🖥️ **Device:** {info['device']}
            📐 **Input Shape:** {info['input_shape']}
            📊 **Output Shape:** {info['output_shape']}
            📝 **Input Names:** {', '.join(info['input_names'])}
            📤 **Output Names:** {', '.join(info['output_names'])}
            💾 **Model Size:** {info['model_size']}
            """)

        except Exception as e:
            return mo.md(f"❌ Error getting model info: {str(e)}")

    return (show_model_info,)


@app.cell
def _(delete_button, mo, model_dropdown, show_model_info):
    mo.vstack([
        mo.vstack([
            mo.hstack([model_dropdown, delete_button], justify="start"),
            show_model_info()
        ])
    ])
    return


@app.cell
def _(mo):
    # Inference input
    inference_input = mo.ui.text_area(
        value="Hello, how are you?", 
        label="📝 Input Text",
        full_width=True
    )

    # Image URL for vision models
    image_url = mo.ui.text(
        value="", 
        label="🖼️ Image URL (for vision models)",
        full_width=True,
        placeholder="https://example.com/image.jpg"
    )

    return image_url, inference_input


@app.cell
def _(
    Image,
    core,
    get_selected_device,
    get_selected_model,
    httpx,
    image_url,
    inference_input,
    io,
    json,
    mo,
    np,
):
    def run_inference():
        if not get_selected_model():
            return "No model selected"

        try:
            model_path = f"models/{get_selected_model()}.xml"
            model = core.read_model(model_path)
            compiled_model = core.compile_model(model, get_selected_device())

            # Prepare input based on model type
            if image_url.value:
                # Vision model
                response = httpx.get(image_url.value)
                image = Image.open(io.BytesIO(response.content))
                image = image.resize((224, 224))  # Standard size
                input_data = np.array(image).astype(np.float32) / 255.0
                input_data = np.expand_dims(input_data, axis=0)
            else:
                # Text model (simplified)
                input_data = np.array([inference_input.value.encode()], dtype=np.object_)

            # Run inference
            results = compiled_model(input_data)

            # Format output
            if isinstance(results, dict):
                output = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in results.items()}
            else:
                output = results.tolist() if hasattr(results, 'tolist') else results

            return f"✅ Inference completed\n\n**Output:**\n```json\n{json.dumps(output, indent=2)}\n```"

        except Exception as e:
            return f"❌ Inference error: {str(e)}"

    inference_button = mo.ui.button(
        label="🚀 Run Inference", 
        kind="success",
        on_change=run_inference
    )

    inference_ui = mo.vstack([inference_input, image_url, inference_button], justify="start")
    return (inference_ui,)


@app.cell
def _(mo):
    # Performance benchmarking
    benchmark_button = mo.ui.button(
        label="⚡ Benchmark Model",
        kind="success",
        on_change=lambda: "Running benchmark..."
    )

    return (benchmark_button,)


@app.cell
def _(benchmark_button, core, get_selected_device, get_selected_model, mo, np):
    def run_benchmark():
        if not get_selected_model():
            return "No model selected"

        try:
            model_path = f"models/{get_selected_model()}.xml"
            model = core.read_model(model_path)
            compiled_model = core.compile_model(model, get_selected_device())

            # Create dummy input
            input_shape = model.inputs[0].shape
            dummy_input = np.random.random(input_shape).astype(np.float32)

            # Warm up
            for _ in range(10):
                compiled_model(dummy_input)

            # Benchmark
            import time
            times = []
            for _ in range(100):
                start = time.time()
                compiled_model(dummy_input)
                times.append(time.time() - start)

            avg_time = np.mean(times) * 1000  # Convert to ms
            fps = 1.0 / np.mean(times)

            return f"""
            **Benchmark Results:**

            ⏱️ **Average Inference Time:** {avg_time:.2f} ms
            🚀 **Throughput:** {fps:.2f} FPS
            📊 **Min Time:** {np.min(times) * 1000:.2f} ms
            📈 **Max Time:** {np.max(times) * 1000:.2f} ms
            """

        except Exception as e:
            return f"❌ Benchmark error: {str(e)}"

    if benchmark_button.value:
        benchmark_result = run_benchmark()
        mo.md(benchmark_result)

    return


@app.cell
def _(mo):
    # Model optimization
    optimization_options = mo.ui.multiselect(
        options=["INT8", "FP16", "CPU_SPARSE_WEIGHTS"],
        label="🔧 Optimization Options"
    )

    optimize_button = mo.ui.button(
        label="⚙️ Optimize Model",
        kind="success",
        on_change=lambda: "Optimizing model..."
    )

    return optimization_options, optimize_button


@app.cell
def _(core, get_selected_model, mo, optimization_options, optimize_button):
    def optimize_model():
        if not get_selected_model():
            return "No model selected"

        try:
            model_path = f"models/{get_selected_model()}.xml"
            model = core.read_model(model_path)

            # Apply optimizations
            if "INT8" in optimization_options.value:
                model = core.quantize_model(model, "INT8")
            if "FP16" in optimization_options.value:
                model = core.convert_model(model, "FP16")

            # Save optimized model
            optimized_path = f"models/{get_selected_model()}_optimized.xml"
            core.save_model(model, optimized_path)

            return f"✅ Model optimized and saved to {optimized_path}"

        except Exception as e:
            return f"❌ Optimization error: {str(e)}"

    if optimize_button.value:
        optimization_result = optimize_model()
        mo.md(optimization_result)

    return


@app.cell
def _(mo):
    # Device information
    device_info_button = mo.ui.button(
        label="ℹ️ Device Info",
        on_change=lambda: "Getting device info..."
    )

    return (device_info_button,)


@app.cell
def _(core, device_info_button, get_devices, mo):
    def get_device_info():
        info = {}
        for device in get_devices():
            try:
                device_info = core.get_property(device, "FULL_DEVICE_NAME")
                info[device] = device_info
            except:
                info[device] = "Device info not available"

        return info

    if device_info_button.value:
    
        info_text = "\n".join([f"**{device}:** {info}" for device, info in get_device_info().items()])
        mo.md(f"**Device Information:**\n\n{info_text}")

    return


@app.cell
def _(mo):
    # Model conversion
    conversion_input = mo.ui.text(
        label="🔄 Convert Model",
        placeholder="Path to model file (.onnx, .pb, .pth, etc.)",
        full_width=True
    )

    convert_button = mo.ui.button(
        label="🔄",
        tooltip="Convert to OpenVINO",
        kind="success",
        on_change=lambda: "Converting model..."
    )

    return conversion_input, convert_button


@app.cell
def _(conversion_input, convert_button, core, mo):
    def convert_model():
        if not conversion_input.value:
            return "No input path provided"

        try:
            input_path = conversion_input.value
            output_path = input_path.rsplit('.', 1)[0] + ".xml"

            # Convert model
            model = core.read_model(input_path)
            core.save_model(model, output_path)

            return f"✅ Model converted and saved to {output_path}"

        except Exception as e:
            return f"❌ Conversion error: {str(e)}"

    if convert_button.value:
        conversion_result = convert_model()
        mo.md(conversion_result)

    return


@app.cell
def _(mo):
    tabs = mo.ui.tabs({
        "📥 Load": "load",
        "🖥️ Devices": "device",
        "🤖 Manage": "manage",
        "🚀 Inference": "inference",
        "⚡ Benchmark": "benchmark",
        "🔧 Optimize": "optimize",
        "🔄 Convert": "convert"
    })
    
    # Main layout
    mo.md(f"""
    # 🤖 OpenVINO Manager
    
    {tabs}
    """)
    return (tabs,)


@app.cell
def _(
    benchmark_button,
    conversion_input,
    convert_button,
    delete_button,
    device_dropdown,
    device_info,
    device_info_button,
    inference_ui,
    load_ui,
    mo,
    model_dropdown,
    optimization_options,
    optimize_button,
    show_model_info,
    tabs,
):
    if tabs.value == "load":
        mo.vstack([mo.md("## 📥 Model Loading"), load_ui])
    elif tabs.value == "device":
        mo.vstack([mo.md("## 🖥️ Device Management"), device_dropdown, device_info, device_info_button])
    elif tabs.value == "manage":
        mo.vstack(
            [
                mo.md("## 🤖 Model Management"),
                mo.hstack([model_dropdown, delete_button], justify="start"),
                show_model_info(),
            ]
        )
    elif tabs.value == "inference":
        mo.vstack([mo.md("## 🚀 Inference"), inference_ui])
    elif tabs.value == "benchmark":
        mo.vstack([mo.md("## ⚡ Performance Benchmarking"), benchmark_button]).callout()
    elif tabs.value == "optimize":
        mo.vstack([mo.md("## 🔧 Model Optimization"), optimization_options, optimize_button])
    elif tabs.value == "convert":
        mo.vstack([mo.md("## 🔄 Model Conversion"), conversion_input, convert_button])
    return


if __name__ == "__main__":
    app.run()
