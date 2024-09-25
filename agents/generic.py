from azure.identity import DeviceCodeCredential, InteractiveBrowserCredential, ClientSecretCredential
from typing import Literal
import os

def openai_creds_ad(method : Literal["DeviceCode", "Interactive", "ClientSecret"] = "Interactive") -> None:
    """
    Retrieve Azure OpenAI API key via Device Code, Interactive, or ClientSecret authentication
    (DeviceCode and Interactive are not suggested for long-running tasks, since key expires every hour)

    :returns: API key assigned to `AZURE_OPENAI_API_KEY` and `OPENAI_API_KEY` environ variables
    """
    if method == "Interactive":
        credential = InteractiveBrowserCredential()
    elif method == "DeviceCode":
        credential = DeviceCodeCredential()
    elif method == "ClientSecret":
        credential = ClientSecretCredential(
            tenant_id=os.environ["SP_TENANT_ID"],
            client_id=os.environ["SP_CLIENT_ID"],
            client_secret=os.environ["SP_CLIENT_SECRET"]
        )
    else:
        raise ValueError("Method must be one of DeviceCode, Interactive, or ClientSecret")
    
    os.environ["AZURE_OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token
    os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]