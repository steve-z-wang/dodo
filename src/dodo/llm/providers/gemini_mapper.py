"""Gemini format conversion utilities."""

import base64
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

from google import genai
from google.genai import types

from dodo.llm.message import (
    Message,
    SystemMessage,
    UserMessage,
    ModelMessage,
)
from dodo.llm.content import Text, Image, ToolResult
from dodo.llm.message import ToolCall

if TYPE_CHECKING:
    from dodo.tools import Tool


def messages_to_gemini_content(
    messages: List[Message],
) -> Tuple[List[types.Content], Optional[str]]:
    """Convert DoDo messages to Gemini's content format.

    Gemini uses alternating user/model roles. System messages are
    extracted separately for use with system_instruction.

    Returns:
        Tuple of (gemini_contents, system_instruction)
    """
    gemini_messages = []
    system_instruction = None

    for msg in messages:
        if isinstance(msg, SystemMessage):
            # Extract system instruction
            if msg.content:
                texts = [p.text for p in msg.content if isinstance(p, Text)]
                system_instruction = "\n\n".join(texts)

        elif isinstance(msg, UserMessage):
            parts = []

            if msg.content:
                for content_part in msg.content:
                    if isinstance(content_part, Text):
                        parts.append(types.Part.from_text(text=content_part.text))
                    elif isinstance(content_part, Image):
                        parts.append(
                            types.Part.from_bytes(
                                data=base64.b64decode(content_part.base64),
                                mime_type=content_part.mime_type.value,
                            )
                        )
                    elif isinstance(content_part, ToolResult):
                        # Tool results go as function responses
                        response_data = {"status": content_part.status.value}
                        if content_part.description:
                            response_data["description"] = content_part.description
                        if content_part.error:
                            response_data["error"] = content_part.error
                        parts.append(
                            types.Part.from_function_response(
                                name=content_part.name,
                                response=response_data,
                            )
                        )

            if parts:
                gemini_messages.append(types.Content(role="user", parts=parts))

        elif isinstance(msg, ModelMessage):
            parts = []

            # Include text content if present
            if msg.content:
                for content_part in msg.content:
                    if isinstance(content_part, Text):
                        parts.append(types.Part.from_text(text=content_part.text))
                    elif isinstance(content_part, Image):
                        parts.append(
                            types.Part.from_bytes(
                                data=base64.b64decode(content_part.base64),
                                mime_type=content_part.mime_type.value,
                            )
                        )

            # Add tool calls as function call parts
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    parts.append(
                        types.Part.from_function_call(
                            name=tool_call.name,
                            args=tool_call.arguments,
                        )
                    )

            if parts:
                gemini_messages.append(types.Content(role="model", parts=parts))

    return gemini_messages, system_instruction


def gemini_response_to_model_message(response) -> ModelMessage:
    """Convert Gemini response to ModelMessage."""
    tool_calls = []
    content_parts = []
    thoughts = None

    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            # Check for text content
            if hasattr(part, "text") and part.text:
                # First text is treated as thoughts
                if thoughts is None:
                    thoughts = part.text
                else:
                    content_parts.append(Text(text=part.text))
            # Check for function call
            elif hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                args = dict(fc.args) if fc.args else {}
                tool_calls.append(
                    ToolCall(
                        name=fc.name,
                        arguments=args,
                    )
                )

    return ModelMessage(
        content=content_parts if content_parts else None,
        tool_calls=tool_calls if tool_calls else None,
        thoughts=thoughts,
    )


def build_tool_config(tools: List["Tool"]) -> types.Tool:
    """Build Gemini Tool with function declarations from tools."""
    function_declarations = []

    for tool in tools:
        # Convert Pydantic model to JSON schema
        params_schema = tool.Params.model_json_schema()

        # Clean schema to be Gemini-compatible
        params_schema = clean_schema_for_gemini(params_schema)

        function_declarations.append(
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=params_schema,
            )
        )

    return types.Tool(function_declarations=function_declarations)


def clean_schema_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Clean Pydantic JSON schema to be compatible with Gemini.

    Gemini only accepts a subset of JSON schema fields and doesn't support $ref.
    This function resolves $ref references and removes unsupported fields.
    """
    # First resolve all $ref references
    schema = resolve_json_schema_refs(schema)

    # Fields that Gemini supports
    allowed_fields = {
        "type",
        "description",
        "enum",
        "items",
        "properties",
        "required",
        "nullable",
        "format",
    }

    cleaned = {}

    for key, value in schema.items():
        if key in allowed_fields:
            if key == "properties" and isinstance(value, dict):
                # Recursively clean nested properties
                cleaned[key] = {
                    prop_name: clean_schema_for_gemini(prop_schema)
                    for prop_name, prop_schema in value.items()
                }
            elif key == "items" and isinstance(value, dict):
                # Recursively clean array items schema
                cleaned[key] = clean_schema_for_gemini(value)
            else:
                cleaned[key] = value
        elif key == "anyOf":
            # Handle anyOf (used for Optional types)
            if isinstance(value, list):
                non_null_schemas = [s for s in value if s.get("type") != "null"]
                if non_null_schemas:
                    non_null_schema = clean_schema_for_gemini(non_null_schemas[0])
                    cleaned.update(non_null_schema)
                    if len(value) > len(non_null_schemas):
                        cleaned["nullable"] = True

    return cleaned


def resolve_json_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve all $ref references in a JSON schema."""
    if "$defs" not in schema and "definitions" not in schema:
        return schema

    defs = schema.get("$defs", schema.get("definitions", {}))

    def resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"]
                # Extract definition name from "#/$defs/Name" or "#/definitions/Name"
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path[8:]
                elif ref_path.startswith("#/definitions/"):
                    def_name = ref_path[14:]
                else:
                    return obj
                if def_name in defs:
                    return resolve(defs[def_name])
                return obj
            return {k: resolve(v) for k, v in obj.items() if k not in ("$defs", "definitions")}
        elif isinstance(obj, list):
            return [resolve(item) for item in obj]
        return obj

    return resolve(schema)
