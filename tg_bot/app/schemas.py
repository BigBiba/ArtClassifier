from pydantic import BaseModel

class PredictionResult(BaseModel):
    class_id: int
    class_name: str
    subclass_id: int
    subclass_name: str