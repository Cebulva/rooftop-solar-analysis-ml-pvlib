# SolarSight

**SolarSight** is an end-to-end web application for automated rooftop solar analysis. It combines deep learning, geospatial processing, and solar physics simulation to help homeowners evaluate their solar potential and connect with local installers.

---

## Overview

SolarSight automates the solar assessment process:

1. **Locate** - Find your building using address search or map selection
2. **Segment** - AI-powered roof detection using a custom U-Net model
3. **Refine** - Interactive canvas for manual boundary adjustments
4. **Analyze** - Calculate solar production using PVGIS satellite data
5. **Report** - Generate PDF reports with financial and environmental metrics
6. **Connect** - Find nearby solar installers and request quotes

---

## Features

- **AI Roof Segmentation**: Custom-trained U-Net model extracts roof structures from satellite imagery
- **Interactive Refinement**: Draw and adjust roof boundaries with precision tools
- **Solar Simulation**: PVGIS-based energy yield calculations with location-specific irradiance data
- **Financial Analysis**: Investment costs, payback period, and 25-year profit projections (German market)
- **Environmental Impact**: CO2 avoided, trees equivalent, and other sustainability metrics
- **Dealer Finder**: Search nearby solar installers with quality/price/delivery scoring
- **Quote System**: In-app quote requests with email notifications
- **AI Chat Assistant**: RAG-powered chatbot for solar-related questions
- **PDF Reports**: Downloadable reports with system specs, financials, and panel visualization

---

## Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Frontend** | Streamlit, Streamlit-Folium, Streamlit-Drawable-Canvas |
| **Deep Learning** | PyTorch, segmentation-models-pytorch (U-Net) |
| **Computer Vision** | OpenCV, Pillow, scikit-image |
| **Solar Simulation** | pvlib, pysolar, PVGIS API |
| **Geospatial** | GeoPandas, Shapely, Folium, geopy, OSMnx |
| **LLM / RAG** | Groq API, ChromaDB, sentence-transformers |
| **Data** | NumPy, Pandas, SQLite |
| **PDF Generation** | fpdf2 |
| **APIs** | PVGIS (EU JRC), OpenStreetMap, Esri ArcGIS Imagery |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/azizisahand/rooftop-solar-analysis-ml-pvlib.git
cd rooftop-solar-analysis-ml-pvlib

# Create conda environment
conda env create -f environment.yml
conda activate solar-env

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (GROQ_API_KEY, etc.)

# Run the application
streamlit run app.py
```

---

## Usage

1. **Start** - Launch the app and click "Start New Analysis"
2. **Locate** - Search for your address or click on the map
3. **Refine** - Adjust the AI-detected roof boundary if needed
4. **Questionnaire** - Enter household details and energy consumption
5. **Analysis** - Review solar panel placement and energy calculations
6. **Report** - Download PDF report and find local installers

---

## Project Structure

```
├── app.py                 # Main Streamlit application
├── stages/                # Multi-step workflow stages
│   ├── stage_0_start.py
│   ├── stage_1_preview.py
│   ├── stage_2_refine.py
│   ├── stage_3a_questionnaire.py
│   ├── stage_3b_solar.py
│   └── stage_4_report.py
├── src/                   # Core modules
│   ├── model_engine.py    # U-Net model loading
│   ├── solar_analysis.py  # PVGIS integration
│   ├── german_solar_calculator.py
│   ├── pdf_generator.py
│   ├── dealer_finder.py
│   ├── rag_bot.py         # AI chat assistant
│   └── ...
├── model/                 # Trained U-Net weights
├── data/                  # Knowledge base for RAG
└── environment.yml        # Conda dependencies
```

---

## Team

| Name | GitHub |
| :--- | :--- |
| Victoria Vasilieva | [@victoria-vasilieva](https://github.com/victoria-vasilieva) |
| Olaf Bulas | [@Cebulva](https://github.com/Cebulva) |
| Sahand Azizi | [@azizisahand](https://github.com/azizisahand) |

---

## License

This project is developed as part of an academic program. See LICENSE for details.
