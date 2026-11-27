import os

import httpx
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.exceptions import ModelAPIError, ModelAuthenticationError, \
    ModelConnectionError, ModelInvalidRequestError, ModelNotFoundError, \
    ModelPermissionDeniedError, ModelRateLimitError, ModelTimeoutError


def create_model(
    model_id: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseChatModel:
    """Initialize the selected large language model.

    Args:
        model_id (str | None, optional): Model identifier, optionally loaded from the environment.
            Defaults to None.
        provider (str | None, optional): Model provider. Defaults to None.
        api_key (str | None, optional): API key for authentication. Defaults to None.
        base_url (str | None, optional): Optional custom model endpoint. Defaults to None.

    Returns:
        BaseChatModel: Initialized language model.
    """
    env_provider = os.getenv("MODEL_PROVIDER")

    model_id = model_id or os.getenv("MODEL_ID")
    provider = provider or env_provider

    if not model_id or not model_id.strip():
        raise ValueError("No model identifier provided. Please select a model.")

    if base_url is None and provider == env_provider:
        base_url = os.getenv("BASE_URL")

    kwargs: dict[str, str] = {}

    if base_url:
        kwargs["base_url"] = base_url

    if provider == "lm-studio":
        provider = "openai"
        kwargs.setdefault("base_url", "http://localhost:1234/v1")
        kwargs["api_key"] = api_key or "lm-studio"

    elif provider != "ollama" and api_key:
        kwargs["api_key"] = api_key

    return init_chat_model(
        model=model_id,
        model_provider=provider,
        configurable_fields=None,
        **kwargs,
    )


def check_model_connection(model: BaseChatModel) -> tuple[bool, str]:
    """Check whether the configured LLM can process a request.

    Returns:
        tuple[bool, str]: A boolean indicating whether the model responded successfully and a
            message describing the result or error.
    """
    try:
        model.invoke("Respond with OK.")
        return True, "Model connection successful."

    except ModelAuthenticationError:
        message = (
            "Authentication error: Your API key could not be verified. Please check that it is "
            "entered correctly and has not expired."
        )

    except ModelPermissionDeniedError:
        message = (
            "Permission error: Your account may not have access to use the selected model. please "
            "check your provider account settings."
        )

    except ModelNotFoundError:
        message = (
            "Model not found: Please check that the selected model name is correct and available "
            "through your provider."
        )

    except ModelInvalidRequestError:
        message = (
            "Invalid request: Your model provider rejected the request. Please check that the "
            "model identifier is correct and that your model configuration is supported."
        )

    except ModelRateLimitError:
        message = (
            "Rate limit exceeded: Your model provider is currently limiting requests. Please "
            "try again later or check your account's usage limits."
        )

    except ModelTimeoutError:
        message = (
            "Request timed out: The model took too long to respond. Please try again or check your "
            "network connection."
        )

    except (ModelConnectionError, httpx.ConnectError):
        message = (
            "Connection error: Could not connect to the model provider. Please check your internet "
            "connection or verify your local model server is running."
        )

    except ModelAPIError:
        message = (
            "Provider error: The model provider encountered an error while processing your "
            "request. Please try again later."
        )

    except Exception as exc: # pylint: disable=broad-exception-caught
        #traceback.print_exc()
        message = (
            "An unexpected error occurred while checking the model connection. Error type: "
            f"{type(exc).__name__}. {exc}"
        )

    return False, message
