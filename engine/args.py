from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class WeatherConfig:
    temperature: int = 70

@dataclass
class SnowfallConfig:
    upper_bound: int = 5
    lower_bound: int = 50

@dataclass
class IncomeConfig:
    minimum: int = 50000

@dataclass
class RiskAffinityConfig:
    weather: float = 1
    snowfall: float = 1
    employment: float = 1
    crime_rate: float = 1
    cost_of_living: float = 1

@dataclass
class WeightsConfig:
    weather: float = 1
    snowfall: float = 1
    employment: float = 1
    crime_rate: float = 1
    cost_of_living: float = 1

@dataclass
class Config:
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    snow: SnowfallConfig = field(default_factory=SnowfallConfig)
    risk_affinity: RiskAffinityConfig = field(default_factory=RiskAffinityConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)

# Hydra configuration
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)