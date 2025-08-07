"""
Using models locally served via huggingface via an OpenAI interface
"""
from multiprocessing import Process

from .openai import OpenAIProvider

try:
    import openai
except ImportError as e:
    raise ImportError(
        f"OpenAI package must be installed to use an transformers provider!\n{str(e)}"
    )

try:
    import transformers
except ImportError as e:
    raise ImportError(f"The transformers package must be installed to use transformers provider!\n{str(e)}")

from transformers.commands.serving import ServeCommand, ServeArguments


class TransformersProvider(OpenAIProvider):
    """
    A provider that spins up a local LLM using HuggingFace transformers serving in a new process
    and uses the OpenAI interface for completions
    """
    serving_process : Process

    def __init__(self, model_name: str, host: str = "localhost", port: int = 8000, **kwargs):
        """
        A provider that spins up a local LLM using HuggingFace transformers serving in a new process
        and uses the OpenAI interface for completions. Note that you'll need pytorch and ample GPU resources
        for this provider, as everything runs locally.

        :param str model_name: The name of a huggingface model instance
        :param str host: The name of the host for the huggingface serving instance
        :param int port: The port number of the host to serve the API
        """

        xformers_kwargs = kwargs
        xformers_kwargs.update({"host": host, "port": port})
        self.xformers_kwargs = ServeArguments(**xformers_kwargs)

        base_url = f"http://{host}:{port}/v1"
        super().__init__(model_name, base_url=base_url, api_key="n/a")

    @staticmethod
    def _serving(args: ServeArguments):
        """
        Serve command that runs in separate process
        """
        serve_process = ServeCommand(args)
        serve_process.run()

    def __aenter__(self):
        """
        Start the serving process
        """
        self.serving_process = Process(target=self._serving, name="Transformers-Serving", args=[self.xformers_kwargs])
        self.serving_process.start()
        return super().__aenter__()
    
    def __aexit__(self, exc_type, exc_value, traceback):
        """
        End serving process
        """
        self.serving_process.terminate()
        self.serving_process.join()
        return super().__aexit__(exc_type, exc_value, traceback)