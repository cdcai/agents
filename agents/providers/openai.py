import logging
import os
from typing import List, Type, Union

import backoff
import openai
from azure.identity import ClientSecretCredential, InteractiveBrowserCredential
from openai.types.chat import ChatCompletionMessage

from ..abstract import _Agent, _Provider

logger = logging.getLogger(__name__)

class AzureOpenAIProvider(_Provider):
    """
    An Azure OpenAI Provider for language Agents.

    This provider generally assumes you already have all required environment variables
    set correctly, or will provide them as kwargs which will be passed to AsyncAzureOpenAI at init

    Namely:
    - api_version or OPENAI_API_VERSION
    - azure_endpoint or AZURE_OPENAI_ENDPOINT

    AZURE_OPENAI_API_KEY will be assigned via authentication (either by ClientSecret or Interactive AD Auth depending on `interactive`)
    
    :param str model_name: Model name from the deployments list to use
    :param bool interactive: Should authentication use an Interactive AD Login (T), or ClientSecret (F)?
    :param **kwargs: Any additional kw-args for AsyncAzureOpenAI
    """
    def __init__(self, model_name: str, interactive: bool, **kwargs):
        self.model_name = model_name
        self.interactive = interactive
        self.authenticate()
        self.llm = openai.AsyncAzureOpenAI(**kwargs)

    def authenticate(self) -> None:
        """
        Retrieve Azure OpenAI API key via Interactive AD Login or ClientSecret authentication
        (Interactive is not suggested for long-running tasks, since key expires every hour)

        :returns: API key assigned to `AZURE_OPENAI_API_KEY` and `OPENAI_API_KEY` environ variables
        """
        credential: Union[InteractiveBrowserCredential, ClientSecretCredential]
        
        if self.interactive:
            credential = InteractiveBrowserCredential()
        else:
            credential = ClientSecretCredential(
                tenant_id=os.environ["SP_TENANT_ID"],
                client_id=os.environ["SP_CLIENT_ID"],
                client_secret=os.environ["SP_CLIENT_SECRET"]
            )

        os.environ["AZURE_OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token
        os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]

        if getattr(self, "llm", None) is not None:
            self.llm.api_key = os.environ["AZURE_OPENAI_API_KEY"]

    @backoff.on_exception(backoff.expo, (openai.APIError, openai.AuthenticationError), max_tries=3)
    async def prompt_agent(self, ag: Type[_Agent], prompt: List[dict[str, str]], **kwargs):
        """
        An async version of the main OAI prompting logic.

        :param ag: The calling agent class
        :param prompt: Either a dict or a list of dicts representing the message(s) to send to OAI model
        :param kwargs: Key word arguments passed to completions.create() call (tool calls, etc.)

        :return: An openAI Choice response object
        """

        # Prompts should be passed as a list, so handle
        # the case where we just passed a single dict
        if isinstance(prompt, dict):
            prompt = [prompt]

        ag.scratchpad += f"--- Input ---------------------------\n"
        ag.scratchpad += "\n".join(msg["content"] for msg in prompt if not isinstance(msg, ChatCompletionMessage))
        ag.scratchpad += "\n-----------------------------------\n"
    
        try:
            res = await self.llm.chat.completions.create(
                messages=prompt, model=self.model_name,
                **kwargs
            )
        except openai.AuthenticationError as e:
            logger.info("Auth failed, attempting to re-authenticate before retrying")
            self.authenticate()
            raise e
        except Exception as e:
            # TODO: some error handling here
            logger.debug(e)
            raise e

        out = res.choices[0]

        ag.scratchpad += "--- Output --------------------------\n"
        ag.scratchpad += "Message:\n"
        ag.scratchpad += out.message.content if out.message.content else "<None>" + "\n"

        if len(ag.TOOLS):
            # attempt to parse tool call arguments
            # BUG: OpenAI sometimes doesn't return a "tool_calls" reason and uses "stop" instead. Annoying.
            if out.finish_reason == "tool_calls" or (out.finish_reason == "stop" and len(out.message.tool_calls)):
                out.finish_reason = "tool_calls"
                # Append GPT response to next payload
                # NOTE: This has to come before the next step of parsing
                ag.tool_res_payload.append(out.message)
        
        ag.scratchpad += "\n-----------------------------------\n"
        logger.info(f"Received response: {out.message.content}")

        if out.finish_reason == "length":
            ag.truncated = True
            ag.scratchpad += "Response returned truncated from OpenAI due to token length.\n"
            logger.warning("Message returned truncated.")
        return out
    
class OpenAIProvider(AzureOpenAIProvider):
    """
    Standard (non-Azure) OpenAI provider

    Requires `api_key` passed as a kwarg, or OPENAI_API_KEY set as an environment variable
    """
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.authenticate()
        self.llm = openai.AsyncOpenAI(**kwargs)
    
    def authenticate(self):
        pass