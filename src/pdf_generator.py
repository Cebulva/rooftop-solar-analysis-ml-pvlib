"""
PDF Report Generator for Rooftop Solar Analysis
Generates downloadable PDF reports with images and metrics.
"""
from fpdf import FPDF
from io import BytesIO
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List
from PIL import Image


class SolarReportPDF(FPDF):
    """Custom PDF class with header/footer for solar reports."""

    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_margins(15, 20, 15)  # left, top, right margins
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        """Add header to each page."""
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 8, 'Solar Panel Installation Analysis Report', 0, 1, 'C')
        self.line(15, 26, 195, 26)
        self.ln(3)

    def footer(self):
        """Add footer with page numbers."""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def bgr_to_rgb_bytes(bgr_image: np.ndarray, format: str = 'PNG') -> bytes:
    """
    Convert BGR numpy array to RGB image bytes.

    Args:
        bgr_image: OpenCV BGR image as numpy array
        format: Output format ('PNG' or 'JPEG')

    Returns:
        Image as bytes
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)

    # Save to BytesIO
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.getvalue()


def generate_solar_report_pdf(
    solar_results: Dict[str, Any],
    consumption_inputs: Dict[str, Any],
    final_analysis: Dict[str, Any],
    location: Dict[str, Any],
    panel_image: Optional[np.ndarray] = None,
    monthly_data: Optional[Any] = None,
    inquiry_id: Optional[str] = None,
    env_metrics: Optional[Dict[str, float]] = None,
) -> bytes:
    """
    Generate a complete PDF report for solar panel analysis.

    Args:
        solar_results: Dictionary from stage 3b with panel/roof data
        consumption_inputs: Dictionary from stage 3a with consumption data
        final_analysis: Dictionary from GermanSolarCalculator
        location: Dictionary with lat/lon coordinates
        panel_image: BGR numpy array of panel placement visualization
        monthly_data: DataFrame with monthly production data
        inquiry_id: Optional inquiry ID for reference
        env_metrics: Optional environmental impact metrics (CO2, trees, etc.)

    Returns:
        PDF file as bytes
    """
    pdf = SolarReportPDF()
    pdf.add_page()

    # Available width after margins (210 - 15 - 15 = 180mm)
    available_width = 180

    # Title and date
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Rooftop Solar Analysis Report', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')

    # Inquiry ID
    if inquiry_id:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, f'Reference ID: {inquiry_id}', 0, 1, 'C')
    pdf.ln(3)

    # Location info
    lat = location.get('lat', 'N/A')
    lon = location.get('lon', 'N/A')
    pdf.set_font('Helvetica', 'I', 9)
    pdf.cell(0, 5, f'Location: {lat}, {lon}', 0, 1, 'C')
    pdf.ln(8)

    # Extract data with safe defaults
    panel_count = solar_results.get('panel_count', 0)
    system_kwp = solar_results.get('system_kwp', 0)
    total_roof_area = solar_results.get('total_roof_area_m2', 0)
    usable_roof_area = solar_results.get('usable_roof_area_m2', 0)
    azimuth = solar_results.get('azimuth', 0)
    tilt = solar_results.get('tilt_angle', 0)
    roof_form = solar_results.get('roof_form', 'Unknown')
    panel_orientation = solar_results.get('panel_orientation', 'Portrait')

    annual_consumption = consumption_inputs.get('annual_kwh', 0)

    # Calculate production from final_analysis
    specific_yield = final_analysis.get('production', {}).get('specific_yield_kwh_kwp', 900)
    solar_production = system_kwp * specific_yield

    # Coverage
    coverage_pct = (solar_production / annual_consumption * 100) if annual_consumption > 0 else 0

    # Financial calculations
    investment_cost = system_kwp * 1400  # EUR per kWp
    self_consumption_rate = 0.30
    electricity_price = 0.40  # EUR/kWh
    feed_in_tariff = 0.082  # EUR/kWh

    self_consumed = min(solar_production * self_consumption_rate, annual_consumption)
    savings = self_consumed * electricity_price
    fed_in = solar_production - self_consumed
    feed_in_earnings = fed_in * feed_in_tariff
    annual_benefit = savings + feed_in_earnings
    payback_years = investment_cost / annual_benefit if annual_benefit > 0 else 99
    profit_25_year = (annual_benefit * 25) - investment_cost

    # Section 1: System Overview
    _add_section_header(pdf, '1. System Overview', available_width)
    _add_key_value_table(pdf, [
        ('System Size', f'{system_kwp:.2f} kWp'),
        ('Number of Panels', f'{panel_count} x 440W'),
        ('Panel Orientation', str(panel_orientation)),
        ('Total Roof Area', f'{total_roof_area:.1f} m2'),
        ('Usable Roof Area', f'{usable_roof_area:.1f} m2'),
    ], available_width)
    pdf.ln(6)

    # Section 2: Roof Configuration
    _add_section_header(pdf, '2. Roof Configuration', available_width)
    _add_key_value_table(pdf, [
        ('Azimuth (Orientation)', f'{azimuth:.0f} degrees'),
        ('Tilt Angle', f'{tilt:.0f} degrees'),
        ('Roof Type', str(roof_form)),
    ], available_width)
    pdf.ln(6)

    # Section 3: Energy Analysis
    _add_section_header(pdf, '3. Energy Analysis', available_width)
    _add_key_value_table(pdf, [
        ('Annual Consumption', f'{annual_consumption:,.0f} kWh'),
        ('Est. Solar Production', f'{solar_production:,.0f} kWh/year'),
        ('Specific Yield', f'{specific_yield:.0f} kWh/kWp'),
        ('Coverage', f'{coverage_pct:.0f}%'),
    ], available_width)
    pdf.ln(6)

    # Section 4: Financial Analysis
    _add_section_header(pdf, '4. Financial Analysis', available_width)
    _add_key_value_table(pdf, [
        ('Estimated Investment', f'{investment_cost:,.0f} EUR'),
        ('Annual Electricity Savings', f'{savings:,.0f} EUR'),
        ('Annual Feed-in Earnings', f'{feed_in_earnings:,.0f} EUR'),
        ('Total Annual Benefit', f'{annual_benefit:,.0f} EUR'),
        ('Payback Period', f'{payback_years:.1f} years'),
        ('25-Year Net Profit', f'{profit_25_year:,.0f} EUR'),
    ], available_width)
    pdf.ln(6)

    # Section 5: Environmental Impact
    if env_metrics:
        _add_section_header(pdf, '5. Environmental Impact', available_width)
        _add_key_value_table(pdf, [
            ('CO2 Avoided (Annual)', f"{env_metrics.get('co2_avoided_tonnes', 0):.1f} tonnes"),
            ('Trees Equivalent', f"{env_metrics.get('trees_equivalent', 0):.0f} trees"),
            ('Cars Off Road', f"{env_metrics.get('cars_equivalent', 0):.1f} cars/year"),
            ('Coal Saved (Annual)', f"{env_metrics.get('coal_avoided_kg', 0):,.0f} kg"),
        ], available_width)
        pdf.ln(6)

    # Section 6: Panel Placement Image
    if panel_image is not None:
        pdf.add_page()
        _add_section_header(pdf, '6. Panel Placement Visualization', available_width)
        pdf.set_font('Helvetica', '', 9)
        pdf.cell(0, 5, 'Solar panels on roof (red arrow = azimuth, white = north)', 0, 1, 'L')
        pdf.ln(2)
        _add_image_from_array(pdf, panel_image, width=160)
        pdf.ln(6)

    # Section 7: Monthly Breakdown (if available)
    if monthly_data is not None:
        try:
            if len(monthly_data) > 0:
                pdf.add_page()
                _add_section_header(pdf, '7. Monthly Production Estimate', available_width)
                _add_monthly_table(pdf, monthly_data)
        except Exception:
            pass  # Skip monthly table if there's an issue

    # Final notes
    pdf.add_page()
    _add_section_header(pdf, 'Important Notes', available_width)
    pdf.set_font('Helvetica', '', 9)
    notes = [
        'Production estimates based on PVGIS satellite data.',
        'Actual production may vary due to weather and shading.',
        'Financial calculations use current prices and tariffs.',
        'Self-consumption rate of 30% assumed (60-70% with battery).',
        'Installation should be by certified professionals.',
        'All work must comply with local regulations.',
    ]
    for note in notes:
        pdf.cell(0, 5, f'- {note}', 0, 1, 'L')

    pdf.ln(8)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 5, 'Report generated by Rooftop Solar Analysis Tool', 0, 1, 'C')

    # Output to bytes
    output = BytesIO()
    pdf.output(output)
    output.seek(0)
    return output.getvalue()


def _add_section_header(pdf: FPDF, title: str, width: float):
    """Add a styled section header."""
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(width, 7, title, 0, 1, 'L', fill=True)
    pdf.ln(2)


def _add_key_value_table(pdf: FPDF, items: List[tuple], available_width: float):
    """Add a two-column key-value table."""
    col1_width = available_width * 0.45  # 45% for label
    col2_width = available_width * 0.55  # 55% for value

    for key, value in items:
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(col1_width, 6, str(key)[:30], 0, 0, 'L')  # Truncate long keys
        pdf.set_font('Helvetica', '', 9)
        pdf.cell(col2_width, 6, str(value)[:40], 0, 1, 'L')  # Truncate long values


def _add_image_from_array(pdf: FPDF, bgr_image: np.ndarray, width: int = 160):
    """Add an image from BGR numpy array to PDF."""
    if bgr_image is None:
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 8, '[Image not available]', 0, 1, 'L')
        return

    try:
        # Validate image
        if not isinstance(bgr_image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        if bgr_image.dtype != np.uint8:
            bgr_image = bgr_image.astype(np.uint8)

        # Handle grayscale images
        if len(bgr_image.shape) == 2:
            bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

        image_bytes = bgr_to_rgb_bytes(bgr_image, format='PNG')
        image_buffer = BytesIO(image_bytes)
        image_buffer.name = 'image.png'

        # Add image (fpdf2 handles positioning)
        pdf.image(image_buffer, x=pdf.l_margin, w=width)

    except Exception as e:
        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 8, f'[Image error: {str(e)[:50]}]', 0, 1, 'L')
        pdf.set_text_color(0, 0, 0)


def _add_monthly_table(pdf: FPDF, monthly_data):
    """Add monthly breakdown table."""
    pdf.set_font('Helvetica', 'B', 9)

    # Header - use smaller widths
    col1, col2, col3 = 40, 60, 60
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(col1, 7, 'Month', 1, 0, 'C', fill=True)
    pdf.cell(col2, 7, 'Energy (kWh/m2)', 1, 0, 'C', fill=True)
    pdf.cell(col3, 7, 'Sunny Days', 1, 1, 'C', fill=True)

    # Data rows
    pdf.set_font('Helvetica', '', 9)

    try:
        # Find the energy column (may use special character ² or regular 2)
        energy_col = None
        sunny_col = None
        for col in monthly_data.columns:
            if 'Energy Output' in col:
                energy_col = col
            if 'Sunny' in col:
                sunny_col = col

        for idx, row in monthly_data.iterrows():
            month_str = str(idx)[:10]  # Truncate month name

            # Get energy value
            energy = 0
            if energy_col and energy_col in row.index:
                energy = row[energy_col]

            # Get sunny days value
            sunny = 0
            if sunny_col and sunny_col in row.index:
                sunny = row[sunny_col]

            pdf.cell(col1, 6, month_str, 1, 0, 'C')
            pdf.cell(col2, 6, f'{float(energy):.1f}', 1, 0, 'C')
            sunny_str = f'{float(sunny):.0f}' if sunny else 'N/A'
            pdf.cell(col3, 6, sunny_str, 1, 1, 'C')
    except Exception:
        pdf.cell(col1 + col2 + col3, 6, 'Data not available', 1, 1, 'C')