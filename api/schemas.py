"""
Pydantic schemas for Flood Risk Prediction API.
Request/response validation with bounds checking.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, Literal
from typing_extensions import Annotated

# Canonical list of 20 expected raw predictors
EXPECTED_RAW_FEATURES = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors",
]

# Feature bounds (based on training data 1st/99th percentiles)
FEATURE_BOUNDS = {
    "MonsoonIntensity": (0.0, 15.0),
    "TopographyDrainage": (0.0, 15.0),
    "RiverManagement": (0.0, 15.0),
    "Deforestation": (0.0, 15.0),
    "Urbanization": (0.0, 15.0),
    "ClimateChange": (0.0, 15.0),
    "DamsQuality": (0.0, 15.0),
    "Siltation": (0.0, 15.0),
    "AgriculturalPractices": (0.0, 15.0),
    "Encroachments": (0.0, 15.0),
    "IneffectiveDisasterPreparedness": (0.0, 15.0),
    "DrainageSystems": (0.0, 15.0),
    "CoastalVulnerability": (0.0, 15.0),
    "Landslides": (0.0, 15.0),
    "Watersheds": (0.0, 15.0),
    "DeterioratingInfrastructure": (0.0, 15.0),
    "PopulationScore": (0.0, 15.0),
    "WetlandLoss": (0.0, 15.0),
    "InadequatePlanning": (0.0, 15.0),
    "PoliticalFactors": (0.0, 15.0),
}


class PredictionRequest(BaseModel):
    """Request schema for single prediction with bounds validation."""
    
    # Dynamically create fields for all 20 features
    MonsoonIntensity: Annotated[float, Field(ge=0.0, le=15.0, description="Precipitation volume, duration, and peak storm surge intensity")]
    TopographyDrainage: Annotated[float, Field(ge=0.0, le=15.0, description="Slope gradient, natural runoff channels, and elevation variance")]
    RiverManagement: Annotated[float, Field(ge=0.0, le=15.0, description="Condition of river embankments, levee maintenance, and canal dredging")]
    Deforestation: Annotated[float, Field(ge=0.0, le=15.0, description="Loss of vegetative soil cover, root binding, and canopy interception")]
    Urbanization: Annotated[float, Field(ge=0.0, le=15.0, description="Paved surface density reducing natural infiltration and groundwater percolation")]
    ClimateChange: Annotated[float, Field(ge=0.0, le=15.0, description="Deviation in historical precipitation regimes and extreme weather events")]
    DamsQuality: Annotated[float, Field(ge=0.0, le=15.0, description="Structural condition, spillway capacity, and silt accumulation in reservoirs")]
    Siltation: Annotated[float, Field(ge=0.0, le=15.0, description="Sediment buildup reducing hydrological conveyance capacity")]
    AgriculturalPractices: Annotated[float, Field(ge=0.0, le=15.0, description="Terracing, soil tilling intensity, and floodplain agriculture")]
    Encroachments: Annotated[float, Field(ge=0.0, le=15.0, description="Unregulated settlements and industrial structures within floodways")]
    IneffectiveDisasterPreparedness: Annotated[float, Field(ge=0.0, le=15.0, description="Gaps in early warning dissemination, evacuation routes, and disaster response")]
    DrainageSystems: Annotated[float, Field(ge=0.0, le=15.0, description="Culvert dimensions, storm sewer networks, and pumping station readiness")]
    CoastalVulnerability: Annotated[float, Field(ge=0.0, le=15.0, description="Susceptibility to tidal surges, storm tides, and sea-level rise")]
    Landslides: Annotated[float, Field(ge=0.0, le=15.0, description="Unstable slopes contributing debris dams and sudden flash flood surges")]
    Watersheds: Annotated[float, Field(ge=0.0, le=15.0, description="Degraded catchment areas unable to retain seasonal surface runoff")]
    DeterioratingInfrastructure: Annotated[float, Field(ge=0.0, le=15.0, description="Lifespan fatigue on bridges, retaining walls, and drainage barriers")]
    PopulationScore: Annotated[float, Field(ge=0.0, le=15.0, description="Concentration of communities in low-lying hazard zones")]
    WetlandLoss: Annotated[float, Field(ge=0.0, le=15.0, description="Destruction of natural retention basins and marsh buffer areas")]
    InadequatePlanning: Annotated[float, Field(ge=0.0, le=15.0, description="Zoning failures and lack of sustainable storm drainage Master Plans")]
    PoliticalFactors: Annotated[float, Field(ge=0.0, le=15.0, description="Delayed maintenance funding and fragmented disaster management coordination")]

    @field_validator(*EXPECTED_RAW_FEATURES, mode='before')
    @classmethod
    def validate_feature_bounds(cls, v, info):
        """Validate feature is within expected bounds."""
        if not isinstance(v, (int, float)):
            raise ValueError(f"Feature must be numeric, got {type(v).__name__}")
        return float(v)

    model_config = ConfigDict(extra="forbid")  # Reject unknown features


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""
    model_config = ConfigDict(protected_namespaces=())

    predicted_class: Literal["Low", "Medium", "High", "UNCERTAIN"] = Field(..., description="Predicted flood risk class")
    confidence: Annotated[float, Field(ge=0.0, le=100.0, description="Prediction confidence percentage")] = Field(..., description="Prediction confidence percentage")
    probabilities: Dict[str, float] = Field(..., description="Calibrated class probabilities")
    decision: Literal["PREDICTED", "UNCERTAIN"] = Field(..., description="Whether prediction was made or withheld")
    model_version: str = Field(..., description="Model version identifier")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""
    predictions: list[PredictionRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""
    results: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"] = "ok"
    model_version: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response schema."""
    detail: str
    error_type: str


# Example valid request for documentation
EXAMPLE_REQUEST = {
    "MonsoonIntensity": 10.0,
    "TopographyDrainage": 8.0,
    "RiverManagement": 7.0,
    "Deforestation": 8.0,
    "Urbanization": 9.0,
    "ClimateChange": 9.0,
    "DamsQuality": 4.0,
    "Siltation": 8.0,
    "AgriculturalPractices": 6.0,
    "Encroachments": 8.0,
    "IneffectiveDisasterPreparedness": 9.0,
    "DrainageSystems": 5.0,
    "CoastalVulnerability": 7.0,
    "Landslides": 6.0,
    "Watersheds": 7.0,
    "DeterioratingInfrastructure": 8.0,
    "PopulationScore": 8.0,
    "WetlandLoss": 7.0,
    "InadequatePlanning": 8.0,
    "PoliticalFactors": 6.0,
}