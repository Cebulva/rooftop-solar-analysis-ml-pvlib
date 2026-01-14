import pandas as pd
import pvlib
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.location import Location
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

def analyze_solar_potential(lat, lon, tilt=20, azimuth=180):
    try:
        # --- 1. ROBUST WEATHER DATA FETCHING ---
        # We assign to a single variable 'res' first to avoid "unpacking" errors
        # if the library returns 2 items, 3 items, or 4 items.
        res = pvlib.iotools.get_pvgis_tmy(
            lat, lon, map_variables=True, timeout=30
        )
        
        # The first item (index 0) is ALWAYS the dataframe we need.
        weather_data = res[0]
        
        weather_data.index.name = "time"
        
        # --- 2. SETUP SYSTEM ---
        location = Location(lat, lon)
        mount = FixedMount(surface_tilt=tilt, surface_azimuth=azimuth)
        array = Array(
            mount=mount,
            module_parameters={'pdc0': 200, 'gamma_pdc': -0.004},
            temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        )
        system = PVSystem(arrays=[array], inverter_parameters={'pdc0': 200, 'eta_inv_nom': 0.96})
        
        # --- 3. RUN MODEL ---
        mc = ModelChain(system, location, aoi_model='physical', spectral_model='no_loss')
        mc.run_model(weather_data)
        
        # --- 4. ANALYZE RESULTS ---
        cs = location.get_clearsky(weather_data.index)
        
        df = pd.DataFrame({
            'ghi': weather_data['ghi'],
            'ghi_clear': cs['ghi'],
            'energy': mc.results.ac
        })
        
        # Avoid division by zero
        df['Kt'] = df['ghi'] / df['ghi_clear'].replace(0, 1) 
        daylight = df['ghi_clear'] > 50
        
        # Monthly Aggregates
        monthly_energy = df['energy'].resample('M').sum() / 1000 # Convert to kWh
        
        # Weather Classification
        # 1. Calculate Daily Average Clearness Index (Kt)
        # We only look at daylight hours (where clear sky > 50 W/m2)
        daylight_mask = df['ghi_clear'] > 50
        daily_kt = df.loc[daylight_mask, 'Kt'].resample('D').mean()
        
        # 2. Define Categories using Standard Meteorological Thresholds
        # Sunny: Kt > 0.7
        # Partly Cloudy: 0.3 < Kt <= 0.7
        # Cloudy: Kt <= 0.3
        
        sunny_days = daily_kt[daily_kt > 0.7].resample('M').count()
        partly_days = daily_kt[(daily_kt > 0.3) & (daily_kt <= 0.7)].resample('M').count()
        cloudy_days = daily_kt[daily_kt <= 0.3].resample('M').count()
        
        # 3. Create the DataFrame
        final_stats = pd.DataFrame({
            'Energy Output (kWh/m²)': monthly_energy.values,
            'Sunny Days': sunny_days.reindex(monthly_energy.index, fill_value=0).values,
            'Partly Cloudy Days': partly_days.reindex(monthly_energy.index, fill_value=0).values,
            'Cloudy Days': cloudy_days.reindex(monthly_energy.index, fill_value=0).values
        }, index=monthly_energy.index.strftime('%B'))
        
        return final_stats, None

    except Exception as e:
        # Return the error message so the app can display it
        return None, str(e)