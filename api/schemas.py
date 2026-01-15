from pydantic import BaseModel, Field
from typing import Literal

class MachineData(BaseModel):
    air_temperature: float = Field(..., description="Air temperature in Kelvin [K]", examples=[298.1])
    process_temperature: float = Field(..., description="Process temperature in Kelvin [K]", examples=[308.6])
    rotational_speed: int = Field(..., description="Rotational speed in rpm", examples=[1551])
    torque: float = Field(..., description="Torque in Nm", examples=[42.8])
    tool_wear: int = Field(..., description="Tool wear time in minutes", examples=[0])
    type: Literal['L', 'M', 'H'] = Field(..., description="Type of machine quality variant")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "air_temperature": 298.1,
                    "process_temperature": 308.6,
                    "rotational_speed": 1551,
                    "torque": 42.8,
                    "tool_wear": 0,
                    "type": "M"
                }
            ]
        }
    }
