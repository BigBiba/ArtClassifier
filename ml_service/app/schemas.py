from pydantic import BaseModel


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    subclass_id: int
    subclass_name: str

class ErrorResponse(BaseModel):
    detail: str