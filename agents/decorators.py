from functools import wraps, partial
from typing import Optional, Union
import pydantic
import logging

logger = logging.getLogger(__name__)

def response_model_handler(func: Optional[type[pydantic.BaseModel]], /, *, expected_len: Optional[int] = None):
    """
    A decorator that wraps a Pydantic BaseModel and returns string output in the event of parsing errors or the correctly parsed model object otherwise.

    :param BaseModel func: The Pydantic BaseModel to wrap
    :param int expected_len: The expected length of args in the parsed output (optional, currently HACK-y since it applies the same value to all args)
    """
    if not func:
        return partial(response_model_handler, expected_len=expected_len)

    @wraps(func)
    def inner_wrapper(*args, **kwargs) -> Union[pydantic.BaseModel, str]:

        try:
            parsed = func(*args, **kwargs)
        except pydantic.ValidationError as err:
            logger.warning(f"Response didn't pass pydantic validation")
            return f"Pydantic validation failed:\n{str(err)}"
        
        if expected_len:
            # Determine if length mismatch exists
            vars_mismatch = [f"{var}: {len(arr)} != {expected_len}" for var, arr in parsed.model_dump().items() if len(arr) != expected_len]
            
            if len(vars_mismatch):
                logger.warning(f"Got the following length mismatches in parsed args:\n{vars_mismatch}")
                return "Length mismatch in return:\n{}".format("\n".join(vars_mismatch))
            
        return parsed

    return inner_wrapper