"""Request/response bodies for the prediction endpoint.

Deliberately array-in/array-out for now: the milestone this belongs to proves
the Python -> C++ -> Python round trip, so the payload is a flat float vector
rather than an encoded image. Image bytes, class names and confidences arrive
with the real model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    input: list[float] = Field(
        ...,
        description="Flattened input tensor, input_elems floats long.",
    )


class PredictionResponse(BaseModel):
    request_id: int = Field(..., description="Scheduler-assigned id for this request.")
    output: list[float] = Field(..., description="Flattened output tensor.")
    latency_ms: float = Field(..., description="Server-side time spent in predict().")
