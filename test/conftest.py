"""Close every helper registered by a test, including primitive-only fixtures."""

import pytest_asyncio

from agents.providers.openai import OpenAIBatchAPIHelper


@pytest_asyncio.fixture(autouse=True)
async def close_registered_batch_helpers(monkeypatch):
    helpers = []
    register = OpenAIBatchAPIHelper.register_provider

    def tracked_register(helper, provider):
        register(helper, provider)
        helpers.append(helper)

    monkeypatch.setattr(OpenAIBatchAPIHelper, "register_provider", tracked_register)
    yield
    for helper in helpers:
        await helper.close()
