from fastapi import APIRouter, status
from pydantic import BaseModel


router = APIRouter(tags=["Health Check"])


class HealthCheck(BaseModel):
    status: str = "OK"


@router.get(
    "/ping",
    summary="Perform a health check",
    response_description="Return a 200 (OK) HTTP status code.",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def ping() -> HealthCheck:
    return HealthCheck()
