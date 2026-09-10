"""Tests for OpenAI provider resource cleanup."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agents.providers.openai import (
    AzureOpenAIBatchProvider,
    AzureOpenAIProvider,
    OpenAIProvider,
)


@pytest.mark.asyncio
async def test_azure_provider_closes_client_and_retained_credential_on_interrupt():
    credential = SimpleNamespace(close=AsyncMock())
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())),
        close=AsyncMock(),
    )

    with (
        patch.dict(
            "os.environ",
            {
                "AZURE_TENANT_ID": "tenant",
                "AZURE_CLIENT_ID": "client",
                "AZURE_CLIENT_SECRET": "secret",
            },
        ),
        patch(
            "agents.providers.openai.ClientSecretCredential",
            return_value=credential,
        ),
        patch(
            "agents.providers.openai.get_bearer_token_provider",
            return_value=AsyncMock(),
        ) as get_token_provider,
        patch(
            "agents.providers.openai.openai.AsyncAzureOpenAI",
            return_value=client,
        ),
        pytest.raises(KeyboardInterrupt),
    ):
        async with AzureOpenAIProvider("gpt-4o", interactive=False) as provider:
            assert provider._credential is credential
            raise KeyboardInterrupt

    get_token_provider.assert_called_once_with(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    client.close.assert_awaited_once_with()
    credential.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_azure_provider_closes_credential_when_client_close_fails():
    error = RuntimeError("client close failed")
    provider = object.__new__(AzureOpenAIProvider)
    provider.llm = SimpleNamespace(close=AsyncMock(side_effect=error))
    provider._credential = SimpleNamespace(close=AsyncMock())

    with pytest.raises(RuntimeError) as exc_info:
        await provider.close()

    assert exc_info.value is error
    provider._credential.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_batch_provider_closes_all_resources_when_handler_close_fails():
    events = []
    error = RuntimeError("handler close failed")

    async def close_handler():
        events.append("handler")
        raise error

    async def close_client():
        events.append("client")

    async def close_credential():
        events.append("credential")

    provider = object.__new__(AzureOpenAIBatchProvider)
    provider.batch_handler = SimpleNamespace(close=AsyncMock(side_effect=close_handler))
    provider.llm = SimpleNamespace(close=AsyncMock(side_effect=close_client))
    provider._credential = SimpleNamespace(
        close=AsyncMock(side_effect=close_credential)
    )

    with pytest.raises(RuntimeError) as exc_info:
        await provider.close()

    assert exc_info.value is error
    assert events == ["handler", "client", "credential"]


@pytest.mark.asyncio
async def test_non_azure_provider_closes_client_without_credential():
    provider = object.__new__(OpenAIProvider)
    provider.llm = SimpleNamespace(close=AsyncMock())

    await provider.close()

    provider.llm.close.assert_awaited_once_with()
