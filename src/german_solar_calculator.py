"""
German Solar Calculator - Consumption Estimation & ROI Analysis
Based on Stromspiegel 2025 & PVGIS Satellite Data
"""

import requests
import math

# --- GERMAN MARKET CONSTANTS (2025/2026) ---
CONSTANTS = {
    'PANEL_POWER_W': 440,           # Modern standard (e.g., Trina/Jinko Glass-Glass)
    'PANEL_AREA_M2': 1.95,          # Approx 1.76m x 1.13m
    'SYSTEM_LOSS_PERCENT': 14,      # Standard conservative loss
    'INSTALLATION_COST_PER_KW': 1400, # Net price (0% VAT applied later)
    'ELECTRICITY_PRICE_DE': 0.40,   # Approx EUR 0.40/kWh (End consumer avg)
    'FEED_IN_TARIFF': 0.08,         # ~8 cents/kWh (EEG Surplus Feed-in)
    'VAT_RATE': 0.00                # 0% VAT (Nullsteuersatz par. 12 Abs. 3 UStG)
}

# --- GERMAN CONSUMPTION PROFILES (Stromspiegel 2025) ---
BASE_LOAD_PROFILES = {
    1: {'apt': 1300, 'house': 2300},
    2: {'apt': 2000, 'house': 3000},
    3: {'apt': 2500, 'house': 3500},
    4: {'apt': 2600, 'house': 4000},
    5: {'apt': 3000, 'house': 5000}
}

ADD_ONS = {
    'water_heater': 1000,  # Electric Instantaneous Heater (Durchlauferhitzer)
    'ev_car': 2500,        # ~15k km/year @ 17kWh/100km
    'heat_pump': 3500      # Modern efficient heat pump (JAZ ~3.5)
}

# Orientation mapping for PVGIS (0=South, -90=East, 90=West)
ORIENTATION_MAP = {
    "South": 0,
    "South-East": -45,
    "East": -90,
    "South-West": 45,
    "West": 90,
    "North": 180
}

# Tilt angle mapping
TILT_MAP = {
    "Flat (0)": 0,
    "Low (15)": 15,
    "Normal (35)": 35,
    "Steep (45)": 45
}


class GermanSolarCalculator:
    """
    Solar calculator optimized for German market conditions.
    Uses PVGIS for accurate solar yield data and Stromspiegel for consumption estimates.
    """

    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon
        self.base_url = "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc"

    def get_pvgis_data(self, tilt, azimuth, kwp):
        """
        Queries PVGIS for specific yield.

        Args:
            tilt: Panel tilt angle in degrees
            azimuth: Panel orientation (0=South, -90=East, 90=West)
            kwp: System size in kWp

        Returns:
            Annual energy production in kWh
        """
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'peakpower': kwp,
            'loss': CONSTANTS['SYSTEM_LOSS_PERCENT'],
            'mountingplace': 'building',
            'angle': tilt,
            'aspect': azimuth,
            'outputformat': 'json'
        }

        try:
            r = requests.get(self.base_url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()['outputs']['totals']['fixed']['E_y']  # Annual kWh
        except Exception as e:
            print(f"PVGIS Connection Error: {e}")
            # Fallback: estimate based on German average (~950 kWh/kWp)
            return kwp * 950

    def estimate_consumption(self, people, building_type, has_water_heater=False,
                            has_ev=False, has_heat_pump=False):
        """
        Calculates consumption based on German 'Stromspiegel' data.

        Args:
            people: Number of people in household (1-6)
            building_type: 'house' or 'apt'
            has_water_heater: Electric water heater present
            has_ev: Electric vehicle present
            has_heat_pump: Heat pump present

        Returns:
            Estimated annual consumption in kWh
        """
        # Clamp people to max 5 for lookup
        p = min(max(people, 1), 5)

        # Base Load
        base = BASE_LOAD_PROFILES[p][building_type]

        # Add-ons
        total = base
        if has_water_heater:
            total += ADD_ONS['water_heater']
        if has_ev:
            total += ADD_ONS['ev_car']
        if has_heat_pump:
            total += ADD_ONS['heat_pump']

        return total

    def recommend_system_size(self, annual_consumption, specific_yield=950):
        """
        Recommends system size to cover annual consumption.

        Args:
            annual_consumption: Annual electricity consumption in kWh
            specific_yield: Expected kWh per kWp (default 950 for Germany)

        Returns:
            Recommended system size in kWp
        """
        return math.ceil(annual_consumption / specific_yield)

    def calculate_economics(self, system_kwp, annual_production, annual_consumption,
                           self_consumption_ratio=0.30):
        """
        German ROI Logic:
        - Self-consumed energy saves electricity cost
        - Exported energy earns feed-in tariff (EEG)

        Args:
            system_kwp: System size in kWp
            annual_production: Annual solar production in kWh
            annual_consumption: Annual household consumption in kWh
            self_consumption_ratio: Fraction of solar used directly (default 0.30 without battery)

        Returns:
            Dictionary with financial analysis
        """
        # Amount of Solar Energy used directly
        self_consumed_kwh = min(annual_production * self_consumption_ratio, annual_consumption)

        # Amount exported to grid (Einspeisung)
        exported_kwh = max(0, annual_production - self_consumed_kwh)

        # Financials
        savings_from_usage = self_consumed_kwh * CONSTANTS['ELECTRICITY_PRICE_DE']
        earnings_from_feedin = exported_kwh * CONSTANTS['FEED_IN_TARIFF']
        total_annual_benefit = savings_from_usage + earnings_from_feedin

        # Cost (0% VAT for residential solar in Germany)
        invest_cost = system_kwp * CONSTANTS['INSTALLATION_COST_PER_KW'] * (1 + CONSTANTS['VAT_RATE'])

        # ROI calculation
        roi_years = invest_cost / total_annual_benefit if total_annual_benefit > 0 else float('inf')

        # 25-year lifetime profit
        lifetime_benefit = total_annual_benefit * 25 - invest_cost

        return {
            "invest_cost": invest_cost,
            "annual_benefit": total_annual_benefit,
            "savings_usage": savings_from_usage,
            "earnings_feedin": earnings_from_feedin,
            "roi_years": roi_years,
            "self_consumed_kwh": self_consumed_kwh,
            "exported_kwh": exported_kwh,
            "lifetime_profit_25y": lifetime_benefit,
            "monthly_benefit": total_annual_benefit / 12
        }

    def full_analysis(self, people, building_type, roof_orientation, roof_tilt,
                     has_water_heater=False, has_ev=False, has_heat_pump=False,
                     custom_roof_area=None):
        """
        Performs complete solar analysis for a household.

        Args:
            people: Number of people in household
            building_type: 'house' or 'apt'
            roof_orientation: Key from ORIENTATION_MAP
            roof_tilt: Key from TILT_MAP
            has_water_heater: Electric water heater present
            has_ev: Electric vehicle present
            has_heat_pump: Heat pump present
            custom_roof_area: Optional custom roof area in m2

        Returns:
            Complete analysis dictionary
        """
        # 1. Estimate consumption
        consumption = self.estimate_consumption(
            people, building_type, has_water_heater, has_ev, has_heat_pump
        )

        # 2. Recommend system size
        recommended_kwp = self.recommend_system_size(consumption)

        # 3. Get PVGIS production data
        tilt_angle = TILT_MAP.get(roof_tilt, 35)
        azimuth = ORIENTATION_MAP.get(roof_orientation, 0)
        production = self.get_pvgis_data(tilt_angle, azimuth, recommended_kwp)

        # 4. Calculate economics
        economics = self.calculate_economics(recommended_kwp, production, consumption)

        # 5. Calculate panel requirements
        num_panels = math.ceil(recommended_kwp * 1000 / CONSTANTS['PANEL_POWER_W'])
        required_roof_area = num_panels * CONSTANTS['PANEL_AREA_M2']

        # 6. Coverage ratio
        coverage_ratio = production / consumption if consumption > 0 else 0

        return {
            'consumption': {
                'annual_kwh': consumption,
                'monthly_avg_kwh': consumption / 12,
                'people': people,
                'building_type': building_type,
                'has_water_heater': has_water_heater,
                'has_ev': has_ev,
                'has_heat_pump': has_heat_pump
            },
            'system': {
                'recommended_kwp': recommended_kwp,
                'num_panels': num_panels,
                'panel_power_w': CONSTANTS['PANEL_POWER_W'],
                'required_roof_area_m2': required_roof_area,
                'roof_orientation': roof_orientation,
                'roof_tilt': roof_tilt
            },
            'production': {
                'annual_kwh': production,
                'monthly_avg_kwh': production / 12,
                'specific_yield_kwh_kwp': production / recommended_kwp if recommended_kwp > 0 else 0
            },
            'economics': economics,
            'coverage': {
                'ratio': coverage_ratio,
                'percent': coverage_ratio * 100,
                'surplus_kwh': max(0, production - consumption),
                'deficit_kwh': max(0, consumption - production)
            },
            'location': {
                'lat': self.lat,
                'lon': self.lon
            }
        }


def get_consumption_breakdown(people, building_type, has_water_heater, has_ev, has_heat_pump):
    """
    Returns detailed breakdown of consumption estimate.
    Useful for displaying to users.
    """
    p = min(max(people, 1), 5)
    base = BASE_LOAD_PROFILES[p][building_type]

    breakdown = {
        'Base consumption': base,
    }

    total = base

    if has_water_heater:
        breakdown['Electric water heater'] = ADD_ONS['water_heater']
        total += ADD_ONS['water_heater']

    if has_ev:
        breakdown['Electric vehicle'] = ADD_ONS['ev_car']
        total += ADD_ONS['ev_car']

    if has_heat_pump:
        breakdown['Heat pump'] = ADD_ONS['heat_pump']
        total += ADD_ONS['heat_pump']

    breakdown['Total'] = total

    return breakdown
