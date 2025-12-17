from dataclasses import dataclass
import datetime
from typing import Dict

@dataclass
class FndDataPoint:
    timestamp: datetime.date
    value: float
    end_date: datetime.date
    fy: int
    fp: str
    form: str
