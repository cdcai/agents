import asyncio
from typing import Optional, Union
from copy import copy
import time
import json
import openai
import pytest
from openai._legacy_response import HttpxBinaryResponseContent
from openai.types import Batch, FileObject
from openai.resources.files import AsyncFiles
from openai.resources.batches import AsyncBatches
from pytest_mock import MockFixture

import agents

# Mock returns
mock_batch_file_ret = ("abc.txt", b"lorem ipsum", "application/json")

mock_file_obj = FileObject(
    id="file_123",
    bytes=1024,
    created_at=0,
    filename=mock_batch_file_ret[0],
    object="file",
    purpose="batch",
    status="uploaded",
)

mock_batch = Batch(
    id="batch_123",
    completion_window="24h",
    created_at=0,
    endpoint="/v1/chat/completions",
    input_file_id=mock_file_obj.id,
    output_file_id=mock_file_obj.id,
    object="batch",
    status="validating",
)

# Simulate a fake batch response body
def gen_response_body(n: int) -> bytes:
    """
    Generates a mock response body for batch completions.
    """
    response_body = [{
        "id": f"batch_{i + 1}",
        "custom_id": f"task-{i + 1}",
        "response": {
            "status_code": 200,
            "request_id": "req_1234",
            "body": {
                "id": "chatcmpl-11111",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Lorem ipsum, dolor sit amet."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 24, "completion_tokens": 15, "total_tokens": 39},
                "system_fingerprint": None,
            },
        },
        "error": None,
    } for i in range(n)]

    # Encode as bytes
    response_body_bytes = "\n".join(json.dumps(res) for res in response_body).encode("utf-8")

    return response_body_bytes

def retrieve_mock_gen(timeout: int = 10):
    """
    Generates a 'stateful' function that simulates
    the API processing the batch request
    """
    t_start = time.time()
    inner_batch = copy(mock_batch)

    async def inner(self, batch_id, **kwargs):
        if time.time() - t_start < timeout:
            inner_batch.status = "in_progress"
            return inner_batch
        else:
            inner_batch.status = "completed"
            return inner_batch

    return inner

class DummyAgent(agents.Agent):
    SYSTEM_PROMPT = "You're an agent."
    BASE_PROMPT = "Do something.\n{batch}"

    def __init__(self, provider=None, **kwargs):
        super().__init__(
            stopping_condition=agents.StopOnStep(1),
            provider=provider,
            **kwargs
        )

def mock_oai(mocker: MockFixture, n_queries: int = 1, batch_timeout: int = 10):
    mock_response = mocker.Mock(spec=HttpxBinaryResponseContent)
    # Generate a mock response body
    response_body_bytes = gen_response_body(n_queries)
    mocker.patch.object(mock_response, "content", response_body_bytes)
    # Patch AsyncAzureOpenAI:
    # - skip __init__
    # - Patch create method of files class
    # - Patch create and retrieve methods of batches
    mocker.patch.object(openai.AsyncAzureOpenAI, "__init__", return_value=None)
    mocker.patch.object(AsyncFiles, "create", return_value=mock_file_obj)
    mocker.patch.object(AsyncFiles, "content", return_value=mock_response)
    mocker.patch.object(AsyncBatches, "create", return_value=mock_batch)
    mocker.patch.object(AsyncBatches, "retrieve", retrieve_mock_gen(batch_timeout))

@pytest.mark.asyncio
async def test_batch_mocking(mocker: MockFixture):
    """
    Test that our mocking of OpenAI still works
    since it's jank and changes to the upstream API might break it.
    """
    mock_oai(mocker)
    aoi = openai.AsyncAzureOpenAI()
    aoi.files = AsyncFiles(aoi)
    aoi.batches = AsyncBatches(aoi)

    # Test file and batch creation endpoint actually return
    # expected objects
    assert (
        await aoi.files.create(file=mock_batch_file_ret, purpose="batch")
        == mock_file_obj
    )
    assert (
        await aoi.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id="file_123",
        )
        == mock_batch
    )

    # Test that our simulation of the batch API works
    # - Should return in_progress followed by completed after a timeout
    outer_batch = copy(mock_batch)
    outer_batch.status = "in_progress"
    assert await aoi.batches.retrieve("batch_123") == outer_batch
    outer_batch.status = "completed"
    await asyncio.sleep(10)
    assert await aoi.batches.retrieve("batch_123") == outer_batch
    
    # Test that batch content comes through ok
    file_content = await aoi.files.content("file_123")
    assert file_content.content == gen_response_body(1)


@pytest.mark.asyncio
async def test_batch_api(mocker: MockFixture):
    # Patch AsyncAzureOpenAI

    mock_oai(mocker)

    aoi = openai.AsyncAzureOpenAI()
    aoi.files = AsyncFiles(aoi)
    aoi.batches = AsyncBatches(aoi)

    provider = agents.AzureOpenAIBatchProvider(
        "random_model",
        quiet=True
    )

    # Patch in mocked OAI API class
    provider.llm = aoi

    # Set an artificially short timeout for testing
    provider.batch_handler.api_timeout = 3

    async with provider:
        ag = DummyAgent(provider=provider, batch="blah")
        await ag()
 
    assert ag.answer == 'Lorem ipsum, dolor sit amet.'



