from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, List
import datetime
from typing import Dict
from enum import Enum

@dataclass
class FndDataPoint:
    timestamp: datetime.date
    value: float
    end_date: datetime.date
    fy: int
    fp: str
    form: str

@dataclass
class TickDataPoint:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    num_trades: int
    vwap: float

class TickField(Enum):
    CLOSE = 'c'
    HIGH = 'h'
    LOW = 'l'
    NUM_TRADES = 'n'
    OPEN = 'o'
    TIMESTAMP = 't'
    VOLUME = 'v'
    VWAP = 'vw'

class DataSource(ABC):
    """Abstract base class for fundamental data sources"""

    @abstractmethod
    def supports_concept(self, concept: str) -> bool:
        """Check if this data source can provide the concept"""
        pass

    @abstractmethod
    def extract_concept(self, concept: str) -> Optional[List[FndDataPoint]]:
        """Extract data points for the concept"""
        pass

    @abstractmethod
    def get_coverage_period(self) -> tuple[str, str]:
        """Return (start_date, end_date) of coverage"""
        pass