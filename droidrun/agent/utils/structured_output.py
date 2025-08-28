from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel

logger = logging.getLogger("droidrun")

def coerce_to_model(schema: Type[BaseModel], data: Dict[str, Any]) -> Optional[BaseModel]:
    """
    Validate and coerce a dict into a Pydantic v2 BaseModel instance.

    Returns None if validation fails.
    """
    try:
        return schema.model_validate(data)
    except Exception as e:
        # Log keys if dict; otherwise log input type to avoid secondary errors
        keys_or_type = (
            sorted(list(data.keys())) if isinstance(data, dict) else f"type={type(data).__name__}"
        )
        logger.exception(
            "coerce_to_model failed: schema=%s, keys=%s, error=%s",
            getattr(schema, "__name__", str(schema)),
            keys_or_type,
            e,
        )
        return None


def model_json_schema(schema: Type[BaseModel]) -> Dict[str, Any]:
    """
    Return the Pydantic v2 JSON schema for the provided model class.
    Useful for prompt hinting or external schema exposure.
    """
    try:
        return schema.model_json_schema()
    except Exception as e:
        logger.exception(
            "model_json_schema failed: schema=%s, error=%s",
            getattr(schema, "__name__", str(schema)),
            e,
        )
        raise


def schema_instruction(schema: Type[BaseModel]) -> str:
    """
    Produce a short instruction string that can be injected into prompts to
    nudge the model to return the required fields.
    """
    try:
        props = schema.model_fields
        required = [name for name, f in props.items() if f.is_required()]
        fields_desc = ", ".join([
            f"{name}:{getattr(f.annotation, '__name__', str(f.annotation))}"
            for name, f in props.items()
        ])
        return (
            "Return a JSON object matching this schema: "
            f"model={schema.__name__}, required={required}, fields=[{fields_desc}]"
        )
    except Exception as e:
        logger.exception(
            "schema_instruction failed: schema=%s, error=%s",
            getattr(schema, "__name__", str(schema)),
            e,
        )
        raise
