from __future__ import annotations

import dataclasses

from fastapi import APIRouter

from app.hardware import detect_hardware, sample_usage

router = APIRouter(tags=["hardware"])


@router.get("/hardware")
def get_hardware() -> dict:
    return dataclasses.asdict(detect_hardware())


@router.get("/hardware/usage")
def get_hardware_usage() -> dict:
    return dataclasses.asdict(sample_usage())
