"""Tests for FlowDeps and the Ollama-Cloud model factory."""

from marimo_flow.agents.deps import DEFAULT_MODELS, FlowDeps, get_model


def test_default_models_cover_all_roles():
    expected_roles = {
        "route",
        "notebook",
        "problem",
        "model",
        "solver",
        "training",
        "mlflow",
        "lead",
    }
    assert set(DEFAULT_MODELS.keys()) == expected_roles


def test_default_role_models():
    assert DEFAULT_MODELS["route"] == "gemma4:31b-cloud"
    assert DEFAULT_MODELS["notebook"] == "qwen3-coder:480b-cloud"
    assert DEFAULT_MODELS["problem"] == "qwen3-coder:480b-cloud"
    assert DEFAULT_MODELS["model"] == "qwen3.5:cloud"
    assert DEFAULT_MODELS["solver"] == "qwen3-coder:480b-cloud"
    assert DEFAULT_MODELS["training"] == "qwen3-coder:480b-cloud"
    assert DEFAULT_MODELS["mlflow"] == "gpt-oss:20b-cloud"
    assert DEFAULT_MODELS["lead"] == "kimi-k2.5:cloud"


def test_get_model_returns_openai_model_with_ollama_base_url():
    model = get_model("route")
    assert model.model_name == "gemma4:31b-cloud"
    assert "11434/v1" in str(model.client.base_url)


def test_get_model_respects_override():
    model = get_model("solver", override="custom:tag")
    assert model.model_name == "custom:tag"


def test_flow_deps_registry_starts_empty():
    deps = FlowDeps()
    assert deps.registry == {}
    assert deps.models == DEFAULT_MODELS


def test_flow_deps_registry_round_trip():
    deps = FlowDeps()
    sentinel = object()
    deps.registry["mlflow:/some/uri"] = sentinel
    assert deps.registry["mlflow:/some/uri"] is sentinel
