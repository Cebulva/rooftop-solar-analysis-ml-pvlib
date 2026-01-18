# ☀️ SolarSight: AI-Powered Roof Analysis & Solar Potential

**SolarSight** is an end-to-end computer vision pipeline designed to automate rooftop solar assessments. By leveraging deep learning and geospatial math, the tool transforms raw satellite imagery into vectorized geometry for precise energy production modeling.

> [!IMPORTANT]
> **🚧 Work In Progress (WIP)**: This project is currently under active development. Core AI and geometric engines are functional, while interactive reporting features are being finalized.

---

## 🔍 Overview
Calculating solar potential manually is time-consuming. **SolarSight** automates this by:
1.  **Projecting** global coordinates to pixel-precise satellite tiles.
2.  **Segmenting** roof structures using a custom-trained U-Net.
3.  **Refining** the geometry through morphological cleaning and spatial filtering.
4.  **Reporting** energy yield based on vectorized area and azimuth.

---

## 👇 Features
 - <details>
    <summary><b>Technical Highlights (Click to expand)</b></summary>
    
    - <b>AI-Powered Rooftop Extraction</b>: Uses a custom-trained U-Net model to segment roof structures from high-resolution satellite imagery.
    - <b>Geometric Vectorization And Analysis</b>: Converts raw AI pixel masks into clean polygons to calculate precise roof area and orientation (azimuth).
    - <b>Intelligent Noise Suppression</b>: Implements advanced geometric and spatial filtering to distinguish target rooftops from streets, sidewalks, and neighboring structures.
    - <b>Human-in-the-Loop Refinement</b>: Integrated Streamlit dashboard allowing users to manually adjust AI-generated vectors for 100% accuracy.
    - <b>Precise Geospatial Centering</b>: Custom Mercator projection logic ensuring sub-pixel centering of images based on specific Latitude And Longitude coordinates.
    <br>
    </details>

---

## 🛠️ Tech Stack

| Category | Tools & Technologies |
| :--- | :--- |
| **Deep Learning** | PyTorch, Torchvision (U-Net) |
| **Computer Vision** | OpenCV, Scipy (Morphological Filtering) |
| **Frontend/App** | Streamlit, Streamlit-Drawable-Canvas |
| **Data & APIs** | ArcGIS World Imagery API, NumPy, Requests |
| **Geospatial** | Web Mercator Projection, Lat/Lon-to-Pixel Mapping |

---

## 🚀 Roadmap (WIP Status)

- [x] **Core AI Model**: U-Net implementation for roof segmentation.
- [x] **Geospatial Engine**: Pixel-precise centering and multi-tile stitching.
- [x] **Geometry Pipeline**: Object filtering based on area and aspect ratio.
- [ ] **Interactive Canvas**: Manual vertex adjustment in Streamlit (In Development).
- [ ] **Solar Engine**: Calculation of regional irradiation and panel capacity.
- [ ] **Export Logic**: PDF report generation for homeowners.

---
