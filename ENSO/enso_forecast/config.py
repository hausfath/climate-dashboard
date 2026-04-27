"""Configuration for ENSO forecast tool: URLs, API keys, constants, model metadata."""

import os
from pathlib import Path

# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FORECASTS_DIR = DATA_DIR / "forecasts"
OBSERVED_DIR = DATA_DIR / "observed"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Ensure directories exist
for d in [RAW_DIR, FORECASTS_DIR, OBSERVED_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- CDS API Key (Copernicus) ---
# Set via environment variable or edit directly here (do NOT commit real keys)
CDS_API_KEY = os.environ.get("CDS_API_KEY", "")
CDS_API_URL = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")

# --- Source URLs ---
IRI_URL = "https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/"

# CFS URLs: dataInd1/2/3 correspond to earliest/middle/latest 10-day
# initialization windows within the last 30 days. E3 is the most current.
# The /products/CFSv2/ path is the live-updated operational version.
CFS_URLS = [
    "https://www.cpc.ncep.noaa.gov/products/CFSv2/dataInd1/nino34Mon.nc",
    "https://www.cpc.ncep.noaa.gov/products/CFSv2/dataInd2/nino34Mon.nc",
    "https://www.cpc.ncep.noaa.gov/products/CFSv2/dataInd3/nino34Mon.nc",
]
# URL for E3 only (preferred for most current forecast)
CFS_E3_URL = "https://www.cpc.ncep.noaa.gov/products/CFSv2/dataInd3/nino34Mon.nc"

# Relative Niño 3.4 (rONI) — same three rolling windows, member-level
RNINO34_URLS = [
    "https://www.cpc.ncep.noaa.gov/products/CFSv2/dataInd1/rnino34Mon.nc",
    "https://www.cpc.ncep.noaa.gov/products/CFSv2/dataInd2/rnino34Mon.nc",
    "https://www.cpc.ncep.noaa.gov/products/CFSv2/dataInd3/rnino34Mon.nc",
]
RNINO34_E3_URL = "https://www.cpc.ncep.noaa.gov/products/CFSv2/dataInd3/rnino34Mon.nc"

NMME_BASE_URL = "https://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom/ENSMEAN/"

OBSERVED_SSTOI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
OBSERVED_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
OBSERVED_RONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt"

# --- CanSIPS (MSC Datamart) ---
CANSIPS_GRIB_BASE = "https://dd.weather.gc.ca/today/model_cansips/100km/forecast"
CANSIPS_CSV_BASE = "https://dd.weather.gc.ca/today/ensemble/cansips/csv/indices/forecast/monthly"

# --- Nino3.4 Region ---
NINO34_LAT_BOUNDS = (-5.0, 5.0)
NINO34_LON_BOUNDS_180 = (-170.0, -120.0)  # -180 to 180 convention
NINO34_LON_BOUNDS_360 = (190.0, 240.0)    # 0 to 360 convention

# --- Validation ---
VALID_NINO34_RANGE = (-4.0, 4.0)
WARNING_NINO34_RANGE = (-3.0, 3.0)

# --- Season-to-center-month mapping ---
# Each 3-month season maps to its center month number (1-12)
SEASON_TO_CENTER_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
    "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
    "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}

# --- IRI Model Metadata ---
IRI_DYNAMICAL_MODELS = {
    "AUS-ACCESS", "AUS-RELATIVE", "CMC CANSIP", "COLA CCSM4",
    "CS-IRI-MM", "IOCAS ICM", "JMA", "KMA", "LDEO",
    "MetFRANCE", "NASA GMAO", "NCEP CFSv2", "SINTEX-F", "UKMO",
}

IRI_STATISTICAL_MODELS = {
    "BCC_RZDM", "CPC MRKOV", "CSU CLIPR", "NTU CODA", "TONGJI-ML",
    "UCLA-TCD", "UW PSL-CSLIM", "UW PSL-LIM", "XRO",
}

# --- NMME Active Models ---
NMME_MODELS = [
    "NCEP-CFSv2",
    "ECCC-CanESM5",
    "ECCC-GEM5.2-NEMO",
    "NCAR-CESM1",
    "NCAR-CCSM4",
    "NASA-GEOS-S2S-2",
]

# NMME filename pattern: {MODEL}.tmpsfc.{YYYYMM}.ENSMEAN.anom.nc
NMME_FILENAME_TEMPLATE = "{model}.tmpsfc.{yyyymm}.ENSMEAN.anom.nc"

# --- C3S Model Configuration ---
C3S_MODELS = {
    # max_lead_months per ECMWF documentation
    # https://confluence.ecmwf.int/display/CKB/Description+of+the+C3S+seasonal+multi-system
    "ECMWF":        {"system": "51",  "originating_centre": "ecmwf",         "max_lead_months": 7},
    "UKMO":         {"system": "610", "originating_centre": "ukmo",          "max_lead_months": 6},
    "Meteo-France": {"system": "9",   "originating_centre": "meteo_france",  "max_lead_months": 7},
    "DWD":          {"system": "22",  "originating_centre": "dwd",           "max_lead_months": 6},
    "CMCC":         {"system": "4",   "originating_centre": "cmcc",          "max_lead_months": 6},
    "JMA":          {"system": "4",   "originating_centre": "jma",           "max_lead_months": 6},
    "ECCC":         {"system": "5",   "originating_centre": "eccc",          "max_lead_months": 7},
    "NCEP":         {"system": "2",   "originating_centre": "ncep",          "max_lead_months": 7},
    "BOM":          {"system": "2",   "originating_centre": "bom",           "max_lead_months": 7},
}

C3S_DATASET = "seasonal-postprocessed-single-levels"
C3S_VARIABLE = "sea_surface_temperature_anomaly"
C3S_ANOMALY_BASE_PERIOD = "1993-2016"

# --- Model Overlap / Deduplication ---
# Maps (source, model_name) → canonical physical model name
# When computing multi-model means, keep only one per canonical model
MODEL_CANONICAL_MAP = {
    ("CFS", "CFSv2"): "NCEP-CFSv2",
    ("IRI", "NCEP CFSv2"): "NCEP-CFSv2",
    ("NMME", "NCEP-CFSv2"): "NCEP-CFSv2",
    ("C3S", "NCEP"): "NCEP-CFSv2",
    ("IRI", "UKMO"): "UKMO",
    ("C3S", "UKMO"): "UKMO",
    ("IRI", "JMA"): "JMA",
    ("C3S", "JMA"): "JMA",
    ("IRI", "AUS-ACCESS"): "BOM-ACCESS",
    ("C3S", "BOM"): "BOM-ACCESS",
    ("IRI", "MetFRANCE"): "Meteo-France",
    ("C3S", "Meteo-France"): "Meteo-France",
    ("IRI", "NASA GMAO"): "NASA-GEOS",
    ("NMME", "NASA-GEOS-S2S-2"): "NASA-GEOS",
    ("IRI", "CMC CANSIP"): "ECCC-CanESM",
    ("NMME", "ECCC-CanESM5"): "ECCC-CanESM5",
    ("CanSIPS", "CanSIPS-CanESM5"): "ECCC-CanESM5",
    ("NMME", "ECCC-GEM5.2-NEMO"): "ECCC-GEM-NEMO",
    ("CanSIPS", "CanSIPS-GEM-NEMO"): "ECCC-GEM-NEMO",
    ("C3S", "ECCC"): "ECCC-CanSIPS",
}

# Preferred source per canonical model for deduplication
# Priority: CFS plume > C3S > NMME > IRI (per plan)
DEDUP_PREFERENCE = {
    "NCEP-CFSv2": ("CFS", "CFSv2"),
    "UKMO": ("C3S", "UKMO"),
    "JMA": ("C3S", "JMA"),
    "BOM-ACCESS": ("C3S", "BOM"),
    "Meteo-France": ("C3S", "Meteo-France"),
    "NASA-GEOS": ("NMME", "NASA-GEOS-S2S-2"),
    "ECCC-CanESM5": ("CanSIPS", "CanSIPS-CanESM5"),
    "ECCC-GEM-NEMO": ("CanSIPS", "CanSIPS-GEM-NEMO"),
}

# --- HTTP request settings ---
REQUEST_TIMEOUT = 60  # seconds
REQUEST_HEADERS = {
    "User-Agent": "ENSO-Forecast-Tool/1.0 (climate research; Python/requests)"
}
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
