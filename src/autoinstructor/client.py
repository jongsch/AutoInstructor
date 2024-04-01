import json

import instructor
from patch import patch
from instructor import IterableModel
from instructor.dsl.validators import Validator
import openai
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.functional_validators import AfterValidator
from enum import Enum

from typing import List, Annotated, Callable, Optional, TypeVar
import pprint
from joblib import Memory


openai_url = "http://127.0.0.1:8004/v1"

api_key = 'sk-1234'
open_ai_api_client = patch(
    OpenAI(base_url=openai_url, api_key=api_key),
    mode=instructor.Mode.JSON_SCHEMA,
)

memory = Memory("cachedir")

# def stream_response(prompt):
#     response = client.chat.completions.create(
#         messages=[
#             {'role': 'system', 'content':'You are a helpful assistant.'},
#             {'role': 'user', 'content': prompt},
#         ],
#         model="gpt-3.5-turbo",
#         stream=True,
#     )
#     if not response:
#         return
#     for chunk in response:
#         yield(chunk.choices[0].delta.content)

# for streaming_output in stream_response(prompt):
#     print(streaming_output, end='', flush=True)

class LLMClient:
    def __init__(self, client: OpenAI):
        self.client = client

    def create(self, messages, response_model, **kwargs):
        return self.client.chat.completions.create(
            messages=messages,
            response_model=response_model,
            **kwargs,
        )


client = LLMClient(open_ai_api_client)

sys_prompt_json_template = "Please answer in JSON. Here's the json schema you must adhere to:\n<schema>\n{schema}\n</schema>"

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)



# @memory.cache
def conjure_model(
        client: LLMClient,
        messages: list[dict[str, str]],
        response_model: ResponseModelT,
        validation_context: dict | None = None,
        max_retries: int = 10,
        max_tokens: int = 128_000,
        **kwargs,
) -> ResponseModelT:
    response_model_instance: response_model = client.create(
        # messages=messages + [{'role': 'system', 'content': sys_prompt_json_template.format(schema=response_model.json_schema())}],
        model="models/hermes-2-pro-mistral-7b-mlc-q0f16",
        messages=messages,
        response_model=response_model,
        validation_context=validation_context,
        max_retries=max_retries,
        max_tokens=max_tokens,
        override_kwargs={"response_format": {"type": "json_object", "schema": json.dumps(response_model.model_json_schema())}},
        **kwargs,
    )
    return response_model_instance


def conjure_models_streaming(
        client: LLMClient,
        messages: list[dict[str, str]],
        response_model: ResponseModelT,
        validation_context: dict | None = None,
        max_retries: int = 3,
        max_tokens: int = 4096,
        **kwargs,
) -> IterableModel(ResponseModelT):
    response_model_instance: IterableModel(ResponseModelT) = client.create(
        messages=messages,
        response_model=response_model,
        validation_context=validation_context,
        max_retries=max_retries,
        max_tokens=max_tokens,
        **kwargs,
    )
    return response_model_instance


def conjure_model_partial_streaming(
        client: LLMClient,
        messages: list[dict[str, str]],
        response_model: ResponseModelT,
        validation_context: dict | None = None,
        max_retries: int = 3,
        max_tokens: int = 4096,
        **kwargs,
) -> ResponseModelT:
    response_model_instance: ResponseModelT = client.create(
        messages=messages,
        response_model=instructor.Partial[response_model],
        validation_context=validation_context,
        max_retries=max_retries,
        max_tokens=max_tokens,
        **kwargs,
    )
    return response_model_instance



def llm_validator(
    statement: str,
    allow_override: bool = False,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
    llm_client: Optional[LLMClient] = client,
) -> Callable[[str], str]:
    llm_client = llm_client if llm_client else client
    def llm(v: str) -> str:
        resp = llm_client.chat.completions.create(
            response_model=Validator,
            messages=[
                {
                    "role": "system",
                    "content": sysprompt,

                },
                {
                    "role": "user",
                    "content": f"Does `{v}` follow the rules: {statement}",
                },
            ],
            model=model,
            temperature=temperature,
        )  # type: ignore[all]

        # If the response is  not valid, return the reason, this could be used in
        # the future to generate a better response, via reasking mechanism.
        assert resp.is_valid, resp.reason

        if allow_override and not resp.is_valid and resp.fixed_value is not None:
            # If the value is not valid, but we allow override, return the fixed value
            return resp.fixed_value
        return v

    return llm

