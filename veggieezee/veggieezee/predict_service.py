"""
Vegetable Price Prediction Service

Core prediction engine that combines historical data with live market prices
to forecast vegetable prices for Nepal's Kalimati Market.
"""
import os
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
ML_DATA_PATH = BASE_DIR / 'predict' / 'ml' / 'data.xlsx'

_model = None
_label_encoder = None
_feature_cols = None
_historical_data = None
_vegetables_list = None

USE_LIVE_DATA = False

# Vegetable name mapping dictionary
# Maps Kalimati API commodity names to training dataset vegetable names
KALIMATI_TO_TRAINING_MAP = {
    # Tomatoes
    'Tomato Big(Indian)': 'Tomato',
    'Tomato Small(Local)': 'Tomato',
    'Tomato Small(Indian)': 'Tomato',
    'Tomato Small(Terai)': 'Tomato',
    
    # Potatoes
    'Potato Red': 'Potato',
    'Potato Red(Indian)': 'Potato',
    
    # Onions
    'Onion Dry (Indian)': 'Onion (Dry)',
    'Onion Green': 'Spring Onion',
    
    # Cabbage
    'Cabbage': 'Cabbage',
    'Cabbage(Local)': 'Cabbage',
    'Red Cabbbage': 'Cabbage',
    
    # Cauliflower
    'Cauli Local': 'Cauliflower',
    'Cauli Local(Jyapu)': 'Cauliflower',
    'Cauli Terai': 'Cauliflower',
    
    # Carrots
    'Carrot(Local)': 'Carrot',
    'Carrot(Terai)': 'Carrot',
    
    # Radish
    'Raddish White(Local)': 'Radish',
    'Raddish White(Hybrid)': 'Radish',
    
    # Brinjal/Eggplant
    'Brinjal Long': 'Eggplant/Brinjal',
    'Brinjal Round': 'Eggplant/Brinjal',
    
    # Beans
    'French Bean(Local)': 'French Bean (Long)',
    'French Bean(Hybrid)': 'French Bean (Long)',
    'French Bean(Rajma)': 'French Bean (Long)',
    'Cow pea(Long)': 'Cowpea/Long Beans',
    'Green Peas': 'Peas (Green)',
    
    # Gourds
    'Bitter Gourd': 'Bitter Gourd',
    'Bottle Gourd': 'Bottle Gourd',
    'Smooth Gourd': 'Sponge Gourd',
    'Pointed Gourd(Terai)': 'Snake Gourd',
    
    # Squash/Pumpkin
    'Pumpkin': 'Pumpkin',
    'Squash(Long)': 'Zucchini',
    'Squash(Round)': 'Zucchini',
    
    # Cucumber
    'Cucumber(Local)': 'Cucumber',
    'Cucumber(Hybrid)': 'Cucumber',
    'Cucumber(LocalCross)': 'Cucumber',
    
    # Leafy Greens
    'Spinach Leaf': 'Spinach',
    'Mustard Leaf': 'Mustard Greens',
    'Brd Leaf Mustard': 'Mustard Greens',
    'Fenugreek Leaf': 'Fenugreek Leaves',
    'Cress Leaf': 'Cress/Watercress',
    'Coriander Green': 'Coriander Leaves',
    'Lettuce': 'Lettuce',
    
    # Root vegetables
    'Sweet Potato': 'Sweet Potato',
    'Yam': 'Yam',
    'Arum': 'Taro Root',
    'Turnip': 'Turnip',
    'Turnip A': 'Turnip',
    
    # Others
    'Okra': 'Okra',
    'Capsicum': 'Capsicum (Bell Pepper)',
    'Ginger': 'Ginger',
    'Garlic Green': 'Garlic Greens',
    'Garlic Dry Nepali': 'Garlic',
    'Garlic Dry Chinese': 'Garlic',
    'Chilli Green': 'Green Chili',
    'Chilli Green(Bullet)': 'Green Chili',
    'Chilli Green(Machhe)': 'Green Chili',
    'Chilli Green(Akbare)': 'Green Chili',
    'Broccoli': 'Broccoli',
    'Celery': 'Celery',
    'Asparagus': 'Asparagus',
    'Bamboo Shoot': 'Bamboo Shoot',
    'Drumstick': 'Drumstick',
    'Christophine': 'Chayote',
    'Mushroom(Kanya)': 'Mushroom',
    'Mushroom(Button)': 'Mushroom',
    'Neuro': 'Fenugreek Leaves',
    'Sword Bean': 'Cluster Beans',
    'Bakula': 'Colocasia Leaves',
}

# List of all vegetables in training dataset (for validation)
TRAINING_VEGETABLES = [
    'Amaranth Greens', 'Asparagus', 'Bamboo Shoot', 'Beetroot', 'Bitter Gourd',
    'Bottle Gourd', 'Broccoli', 'Cabbage', 'Capsicum (Bell Pepper)', 'Carrot',
    'Cauliflower', 'Celery', 'Chayote', 'Cluster Beans', 'Colocasia Leaves',
    'Coriander Leaves', 'Cowpea/Long Beans', 'Cress/Watercress', 'Cucumber',
    'Drumstick', 'Eggplant/Brinjal', 'Fenugreek Leaves', 'French Bean (Long)',
    'Garlic', 'Garlic Greens', 'Ginger', 'Green Beans', 'Green Chili',
    'Green Papaya (for curry)', 'Jackfruit (Young)', 'Lettuce', 'Mushroom',
    'Mustard Greens', 'Okra', 'Onion (Dry)', 'Peas (Green)', 'Potato',
    'Pumpkin', 'Radish', 'Ridge Gourd', 'Snake Gourd', 'Spinach',
    'Sponge Gourd', 'Spring Onion', 'Sweet Potato', 'Taro Root', 'Tomato',
    'Turnip', 'Yam', 'Zucchini'
]


def get_training_vegetable_name(kalimati_name):
    """
    Map Kalimati API vegetable name to training dataset name.
    
    Args:
        kalimati_name: Vegetable name from Kalimati Market API
        
    Returns:
        str: Corresponding training dataset name, or None if no mapping exists
    """
    if not kalimati_name:
        return None
    
    if kalimati_name in KALIMATI_TO_TRAINING_MAP:
        return KALIMATI_TO_TRAINING_MAP[kalimati_name]
    
    kalimati_lower = kalimati_name.lower().strip()
    for k, v in KALIMATI_TO_TRAINING_MAP.items():
        if k.lower() == kalimati_lower:
            return v
    
    for tv in TRAINING_VEGETABLES:
        if tv.lower() == kalimati_lower:
            return tv
    
    return None


def is_predictable(vegetable_name):
    """
    Check if a vegetable has a prediction model available.
    
    Args:
        vegetable_name: Name of the vegetable to check
        
    Returns:
        bool: True if predictable, False otherwise
    """
    training_name = get_training_vegetable_name(vegetable_name)
    return training_name is not None


def get_live_data_service():
    """Import live data service lazily to avoid circular imports"""
    try:
        from . import live_data_service
        return live_data_service
    except ImportError:
        return None


def load_historical_data():
    """Load historical price data from Excel file"""
    global _historical_data, _vegetables_list
    
    if _historical_data is not None:
        return _historical_data
    
    try:
        df = pd.read_excel(ML_DATA_PATH)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', 'vegetable', 'price_npr'])
        df['vegetable'] = df['vegetable'].astype(str).str.strip().str.lower()
        
        if 'vegetable_nepali' in df.columns:
            df['vegetable_nepali'] = df['vegetable_nepali'].astype(str).str.strip()
        else:
            df['vegetable_nepali'] = df['vegetable'].str.title()
        
        if 'region' in df.columns:
            df['region'] = df['region'].astype(str).str.strip()
        else:
            df['region'] = 'Kalimati'
        
        df = df.sort_values('date').reset_index(drop=True)
        _historical_data = df
        
        veg_df = df[['vegetable', 'vegetable_nepali']].drop_duplicates()
        _vegetables_list = veg_df.to_dict('records')
        
        return df
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        return None


def get_vegetables_list():
    """Get list of available vegetables - prefers database data over live API"""
    global _vegetables_list

    # Try to get from database first (faster and more reliable)
    try:
        from prices.models import VegetablePrice
        from datetime import date

        logger.info(f"Fetching vegetables list for {date.today()}")

        # Get latest prices from database
        latest_prices = VegetablePrice.objects.filter(
            date=date.today()
        ).values('commodity_name', 'avg_price', 'min_price', 'max_price', 'commodity_unit')
        
        if latest_prices.exists():
            vegetables = []
            for item in latest_prices:
                # Extract base name and create Nepali mapping
                name = item['commodity_name']
                base_name = name.split('(')[0].strip()
                
                # Simple Nepali name mapping
                nepali_map = {
                    'Tomato': 'Golbheda', 'Potato': 'Aalu', 'Onion': 'Pyaaj',
                    'Carrot': 'Gajar', 'Cabbage': 'Banda', 'Cauli': 'Kauli',
                    'Raddish': 'Mula', 'Brinjal': 'Bhanta', 'Spinach': 'Palungo',
                    'Cucumber': 'Kakro', 'Bitter Gourd': 'Tite Karela',
                    'Bottle Gourd': 'Lauka', 'Pumpkin': 'Pharsi', 'Ginger': 'Aduwa',
                    'Garlic': 'Lasun', 'Chilli': 'Khursani', 'Capsicum': 'Bhede Khursani',
                }
                
                nepali = nepali_map.get(base_name, base_name)
                
                # Check if this vegetable is predictable
                training_name = get_training_vegetable_name(name)
                
                vegetables.append({
                    'vegetable': name,
                    'vegetable_nepali': nepali,
                    'current_price': float(item['avg_price']),
                    'min_price': float(item['min_price']),
                    'max_price': float(item['max_price']),
                    'unit': item['commodity_unit'],
                    'is_predictable': training_name is not None,
                    'training_name': training_name,
                })

            logger.info(f"Loaded {len(vegetables)} vegetables from database ({len([v for v in vegetables if v['is_predictable']])} predictable)")
            return vegetables
    except Exception as e:
        logger.error(f"Error getting vegetables from database: {e}")
    
    # Fallback to live API (with short timeout)
    if USE_LIVE_DATA:
        live_service = get_live_data_service()
        if live_service:
            try:
                live_vegs = live_service.get_live_vegetables_list()
                if live_vegs:
                    return live_vegs
            except Exception as e:
                logger.error(f"Error getting live vegetables: {e}")
    
    # Fallback to local data
    if _vegetables_list is None:
        load_historical_data()
    
    if _vegetables_list:
        return _vegetables_list
    
    # Final fallback to hardcoded list
    return [
        {'vegetable': 'Tomato Big(Indian)', 'vegetable_nepali': 'Golbheda', 'is_predictable': True, 'training_name': 'Tomato'},
        {'vegetable': 'Potato Red', 'vegetable_nepali': 'Aalu', 'is_predictable': True, 'training_name': 'Potato'},
        {'vegetable': 'Onion Dry (Indian)', 'vegetable_nepali': 'Pyaaj', 'is_predictable': True, 'training_name': 'Onion (Dry)'},
        {'vegetable': 'Cabbage', 'vegetable_nepali': 'Banda', 'is_predictable': True, 'training_name': 'Cabbage'},
        {'vegetable': 'Cauli Local', 'vegetable_nepali': 'Kauli', 'is_predictable': True, 'training_name': 'Cauliflower'},
        {'vegetable': 'Carrot(Local)', 'vegetable_nepali': 'Gajar', 'is_predictable': True, 'training_name': 'Carrot'},
        {'vegetable': 'Spinach Leaf', 'vegetable_nepali': 'Palungo', 'is_predictable': True, 'training_name': 'Spinach'},
        {'vegetable': 'Cucumber(Local)', 'vegetable_nepali': 'Kakro', 'is_predictable': True, 'training_name': 'Cucumber'},
        {'vegetable': 'Bitter Gourd', 'vegetable_nepali': 'Tite Karela', 'is_predictable': True, 'training_name': 'Bitter Gourd'},
        {'vegetable': 'French Bean(Local)', 'vegetable_nepali': 'Simi', 'is_predictable': True, 'training_name': 'French Bean (Long)'},
        {'vegetable': 'Raddish White(Local)', 'vegetable_nepali': 'Mula', 'is_predictable': True, 'training_name': 'Radish'},
        {'vegetable': 'Garlic Dry Nepali', 'vegetable_nepali': 'Lasun', 'is_predictable': True, 'training_name': 'Garlic'},
    ]


def get_predictable_vegetables():
    """Get only vegetables that can be predicted (exist in training data)"""
    all_vegetables = get_vegetables_list()
    return [v for v in all_vegetables if v.get('is_predictable', False)]


def get_live_price_for_vegetable(vegetable_name):
    """Get current live price for a vegetable from Kalimati API"""
    if not USE_LIVE_DATA:
        return None
    
    live_service = get_live_data_service()
    if not live_service:
        return None
    
    try:
        return live_service.get_live_price(vegetable_name)
    except Exception as e:
        logger.error(f"Error getting live price: {e}")
        return None


def get_historical_prices(vegetable, days=30, region=None):
    """
    Get historical prices for a vegetable
    Combines database prices (recent/live) with Excel historical data for best accuracy
    """
    all_prices = []
    
    # STEP 1: Try to get recent prices from database first
    try:
        from prices.models import VegetablePrice
        from datetime import date, timedelta
        import pandas as pd
        
        # Get training vegetable name for database lookup
        training_name = get_training_vegetable_name(vegetable)
        search_name = training_name if training_name else vegetable
        
        # Get last 90 days from database (more than needed, will trim later)
        start_date = date.today() - timedelta(days=90)
        
        # Search for this vegetable in database (case-insensitive partial match)
        db_prices = VegetablePrice.objects.filter(
            commodity_name__icontains=search_name.split()[0],  # Match first word
            date__gte=start_date
        ).order_by('date').values('date', 'avg_price')
        
        # Convert to our format with pandas Timestamp for consistency
        for item in db_prices:
            try:
                all_prices.append({
                    'date': pd.Timestamp(item['date']),
                    'price_npr': float(item['avg_price']),
                    'source': 'database'
                })
            except (OSError, ValueError, OverflowError):
                continue

        logger.debug(f"Found {len(all_prices)} prices from database for {search_name}")
    except Exception as e:
        logger.error(f"Error fetching from database: {e}")
    
    # STEP 2: Get historical data from Excel file
    df = load_historical_data()
    
    if df is not None:
        vegetable_lower = vegetable.lower().strip()
        mask = df['vegetable'] == vegetable_lower
        
        if region:
            mask &= df['region'] == region
        
        # Get all historical data from Excel
        hist_df = df[mask].sort_values('date')
        
        # Convert to list
        for _, row in hist_df.iterrows():
            try:
                all_prices.append({
                    'date': pd.Timestamp(row['date']),
                    'price_npr': float(row['price_npr']),
                    'source': 'excel'
                })
            except (OSError, ValueError, OverflowError):
                continue

        logger.debug(f"Found {len(hist_df)} prices from Excel for {vegetable_lower}")

    # STEP 3: Combine and deduplicate (database takes precedence over Excel for same dates)
    if not all_prices:
        return []
    
    # Sort by date
    all_prices.sort(key=lambda x: x['date'])
    
    # Group by date - average Excel prices for same date, database takes precedence
    date_groups = {}
    for price in all_prices:
        date_key = price['date'].strftime('%Y-%m-%d')
        
        if date_key not in date_groups:
            date_groups[date_key] = {'db': [], 'excel': [], 'date': price['date']}
        
        if price['source'] == 'database':
            date_groups[date_key]['db'].append(price['price_npr'])
        else:
            date_groups[date_key]['excel'].append(price['price_npr'])
    
    combined_prices = []
    for date_key, group in date_groups.items():
        if group['db']:
            avg_price = np.mean(group['db'])
        else:
            avg_price = np.mean(group['excel'])
        
        combined_prices.append({
            'date': group['date'],
            'price_npr': float(avg_price)
        })
    
    combined_prices.sort(key=lambda x: x['date'])
    
    # Return only the requested number of days (most recent)
    result = combined_prices[-days:] if len(combined_prices) > days else combined_prices

    logger.debug(f"Returning {len(result)} combined prices (requested {days} days)")
    return result


def _load_xgboost_model():
    """Load the trained XGBoost model and label encoder"""
    global _model, _label_encoder
    
    if _model is not None:
        return _model, _label_encoder
    
    model_path = MODELS_DIR / 'nepal_veg_price_xgboost.pkl'
    encoder_path = MODELS_DIR / 'nepal_veg_label_encoder.pkl'
    
    try:
        _model = joblib.load(model_path)
        _label_encoder = joblib.load(encoder_path)
        logger.info(f"XGBoost model loaded ({_model.n_estimators} trees, {len(_label_encoder.classes_)} vegetables)")
        return _model, _label_encoder
    except Exception as e:
        logger.error(f"Failed to load XGBoost model: {e}")
        return None, None


XGBOOST_FEATURE_COLS = [
    'season_code', 'is_winter', 'is_spring', 'is_summer',
    'is_monsoon', 'is_autumn', 'is_prewin',
    'is_asar', 'is_shrawan', 'is_bhadra', 'monsoon_intensity',
    'is_rice_planting', 'is_rice_harvest', 'is_winter_veg_harvest',
    'supply_score',
    'is_dashain_window', 'is_tihar_window', 'is_holi_window',
    'is_maghe_window', 'is_chhath_window',
    'is_any_festival', 'festival_demand_weight',
    'is_saturday', 'is_monday', 'is_month_start', 'is_month_end',
    'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos',
    'weekofyear_sin', 'weekofyear_cos', 'dayofyear_sin', 'dayofyear_cos',
    'price_pressure_score',
    'price_lag_1', 'price_lag_3', 'price_lag_7', 'price_lag_14', 'price_lag_30',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
    'rolling_std_7', 'rolling_std_14', 'rolling_std_30',
    'rolling_min_7', 'rolling_max_7',
    'rolling_min_30', 'rolling_max_30',
    'price_change_1d', 'price_change_7d', 'price_change_30d',
    'price_range', 'price_range_lag1',
    'name_encoded',
    'year', 'month', 'day', 'dayofweek', 'weekofyear', 'dayofyear',
]


def build_xgboost_features(target_date, vegetable_name, historical_prices, live_price, label_encoder):
    """
    Build the full 62-feature vector required by the XGBoost model.
    Matches the exact feature engineering from the training notebook.
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    m = target_date.month
    d = target_date.day
    row = {}
    
    # Date features
    row['year'] = target_date.year
    row['month'] = m
    row['day'] = d
    row['dayofweek'] = target_date.dayofweek
    row['weekofyear'] = target_date.isocalendar()[1]
    row['dayofyear'] = target_date.timetuple().tm_yday
    
    # Season features
    season_map = {1:1, 2:1, 3:2, 4:2, 5:3, 6:4, 7:4, 8:4, 9:5, 10:6, 11:6, 12:1}
    row['season_code'] = season_map[m]
    row['is_winter'] = int(row['season_code'] == 1)
    row['is_spring'] = int(row['season_code'] == 2)
    row['is_summer'] = int(row['season_code'] == 3)
    row['is_monsoon'] = int(row['season_code'] == 4)
    row['is_autumn'] = int(row['season_code'] == 5)
    row['is_prewin'] = int(row['season_code'] == 6)
    
    # Nepali calendar months
    row['is_asar'] = int(m == 6)
    row['is_shrawan'] = int(m == 7)
    row['is_bhadra'] = int(m == 8)
    
    # Monsoon and supply
    monsoon_map = {1:0, 2:0, 3:0, 4:0, 5:0.5, 6:2, 7:3, 8:2.5, 9:1, 10:0, 11:0, 12:0}
    row['monsoon_intensity'] = monsoon_map[m]
    supply_map = {1:1, 2:1, 3:0, 4:0, 5:-0.5, 6:-1, 7:-1, 8:-0.5, 9:0.5, 10:1, 11:0.5, 12:0}
    row['supply_score'] = supply_map[m]
    row['is_rice_planting'] = int(m in [6, 7])
    row['is_rice_harvest'] = int(m in [10, 11])
    row['is_winter_veg_harvest'] = int(m in [1, 2])
    
    # Festival windows
    row['is_dashain_window'] = int(m == 10 and d <= 20)
    row['is_tihar_window'] = int((m == 10 and d >= 20) or (m == 11 and d <= 5))
    row['is_holi_window'] = int(m == 3 and d <= 15)
    row['is_maghe_window'] = int(m == 1 and 13 <= d <= 15)
    row['is_chhath_window'] = int(m == 11 and d <= 10)
    row['is_any_festival'] = int(
        row['is_dashain_window'] or row['is_tihar_window'] or
        row['is_holi_window'] or row['is_chhath_window']
    )
    row['festival_demand_weight'] = (
        row['is_dashain_window'] * 1.5 + row['is_tihar_window'] * 1.2 +
        row['is_holi_window'] * 0.6 + row['is_chhath_window'] * 0.5
    )
    
    # Day-of-week features
    row['is_saturday'] = int(row['dayofweek'] == 5)
    row['is_monday'] = int(row['dayofweek'] == 0)
    row['is_month_start'] = int(d <= 5)
    row['is_month_end'] = int(d >= 26)
    
    # Cyclical encoding
    row['month_sin'] = np.sin(2 * np.pi * m / 12)
    row['month_cos'] = np.cos(2 * np.pi * m / 12)
    row['dayofweek_sin'] = np.sin(2 * np.pi * row['dayofweek'] / 7)
    row['dayofweek_cos'] = np.cos(2 * np.pi * row['dayofweek'] / 7)
    row['weekofyear_sin'] = np.sin(2 * np.pi * row['weekofyear'] / 52)
    row['weekofyear_cos'] = np.cos(2 * np.pi * row['weekofyear'] / 52)
    row['dayofyear_sin'] = np.sin(2 * np.pi * row['dayofyear'] / 365)
    row['dayofyear_cos'] = np.cos(2 * np.pi * row['dayofyear'] / 365)
    
    # Composite pressure score
    row['price_pressure_score'] = (
        row['monsoon_intensity'] * 0.4 + row['is_dashain_window'] * 1.5 +
        row['is_tihar_window'] * 1.2 + row['is_any_festival'] * 0.4 +
        row['is_winter'] * 0.5 + row['supply_score'] * -1.0
    )
    
    # Lag features from historical prices (use live + database data)
    prices = [p['price_npr'] for p in historical_prices] if historical_prices else []
    
    # If we have live price, append it as the most recent data point
    if live_price and (not prices or prices[-1] != live_price['avg_price']):
        prices.append(live_price['avg_price'])
    
    if len(prices) == 0:
        prices = [50.0]
    
    row['price_lag_1'] = prices[-1] if len(prices) >= 1 else np.mean(prices)
    row['price_lag_3'] = prices[-3] if len(prices) >= 3 else prices[0]
    row['price_lag_7'] = prices[-7] if len(prices) >= 7 else prices[0]
    row['price_lag_14'] = prices[-14] if len(prices) >= 14 else prices[0]
    row['price_lag_30'] = prices[-30] if len(prices) >= 30 else prices[0]
    row['rolling_mean_7'] = float(np.mean(prices[-7:])) if len(prices) >= 7 else float(np.mean(prices))
    row['rolling_mean_14'] = float(np.mean(prices[-14:])) if len(prices) >= 14 else float(np.mean(prices))
    row['rolling_mean_30'] = float(np.mean(prices[-30:])) if len(prices) >= 30 else float(np.mean(prices))
    row['rolling_std_7'] = float(np.std(prices[-7:])) if len(prices) >= 7 else 0
    row['rolling_std_14'] = float(np.std(prices[-14:])) if len(prices) >= 14 else 0
    row['rolling_std_30'] = float(np.std(prices[-30:])) if len(prices) >= 30 else 0
    row['rolling_min_7'] = float(np.min(prices[-7:])) if len(prices) >= 7 else float(np.min(prices))
    row['rolling_max_7'] = float(np.max(prices[-7:])) if len(prices) >= 7 else float(np.max(prices))
    row['rolling_min_30'] = float(np.min(prices[-30:])) if len(prices) >= 30 else float(np.min(prices))
    row['rolling_max_30'] = float(np.max(prices[-30:])) if len(prices) >= 30 else float(np.max(prices))
    row['price_change_1d'] = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 and prices[-2] != 0 else 0
    row['price_change_7d'] = (prices[-1] - prices[-7]) / prices[-7] if len(prices) >= 7 and prices[-7] != 0 else 0
    row['price_change_30d'] = (prices[-1] - prices[-30]) / prices[-30] if len(prices) >= 30 and prices[-30] != 0 else 0
    
    # Price range from live data or historical
    if live_price:
        row['price_range'] = live_price['max_price'] - live_price['min_price']
        row['price_range_lag1'] = row['price_range']
    else:
        row['price_range'] = float(np.max(prices[-7:])) - float(np.min(prices[-7:])) if len(prices) >= 7 else 0
        row['price_range_lag1'] = row['price_range']
    
    # Vegetable identity encoding
    try:
        row['name_encoded'] = label_encoder.transform([vegetable_name])[0]
    except ValueError:
        row['name_encoded'] = -1
    
    return pd.DataFrame([row])[XGBOOST_FEATURE_COLS]


def predict_price(vegetable, target_date, region=None):
    """
    Predict vegetable price using the trained XGBoost model.
    Falls back to rule-based prediction if model unavailable.
    Combines live Kalimati prices with historical data for lag features.
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    original_name = vegetable
    training_name = get_training_vegetable_name(vegetable)
    
    if training_name is None:
        return {
            'success': False,
            'error': f'No prediction model available for "{vegetable}". This vegetable is not in our training dataset.',
            'vegetable': vegetable,
            'date': target_date.strftime('%Y-%m-%d'),
            'suggestion': 'Try selecting a vegetable marked as "Predictable" in the dropdown.',
        }
    
    vegetable_lower = training_name.lower().strip()
    live_price = get_live_price_for_vegetable(original_name)
    historical = get_historical_prices(vegetable_lower, days=60, region=region)
    
    if not historical and not live_price:
        return {
            'success': False,
            'error': f'No historical data found for {training_name}',
            'vegetable': original_name,
            'training_name': training_name,
            'date': target_date.strftime('%Y-%m-%d'),
        }
    
    # Try XGBoost model first
    model, label_encoder = _load_xgboost_model()
    
    if model is not None and label_encoder is not None:
        try:
            X = build_xgboost_features(target_date, training_name, historical, live_price, label_encoder)
            log_prediction = model.predict(X)[0]
            predicted_price = float(np.exp(log_prediction))
            
            # Confidence bounds from rolling stats
            prices = [p['price_npr'] for p in historical] if historical else []
            if live_price:
                prices.append(live_price['avg_price'])
            
            std = float(np.std(prices[-7:])) if len(prices) >= 7 else float(np.std(prices)) if len(prices) > 1 else predicted_price * 0.1
            price_min = max(5, predicted_price - 2 * std)
            price_max = predicted_price + 2 * std
            
            current_price = live_price['avg_price'] if live_price else (prices[-1] if prices else predicted_price)
            
            # Calculate historical summary from real data
            historical_summary = _calculate_historical_summary(historical, live_price)
            
            logger.info(f"XGBoost prediction for {training_name} on {target_date.strftime('%Y-%m-%d')}: Rs.{predicted_price:.2f}")
            
            return {
                'success': True,
                'vegetable': original_name,
                'training_name': training_name,
                'date': target_date.strftime('%Y-%m-%d'),
                'predicted_price': round(predicted_price, 2),
                'price_min': round(price_min, 2),
                'price_max': round(price_max, 2),
                'confidence': 'High' if live_price else 'Medium',
                'is_live': live_price is not None,
                'model': 'XGBoost',
                'historical_summary': historical_summary,
                'factors': {
                    'base_price': round(current_price, 2),
                    'model_type': 'XGBoost Regressor (2746 trees)',
                    'features_used': len(XGBOOST_FEATURE_COLS),
                    'historical_days': len(historical),
                    'live_data': live_price is not None,
                },
                'recent_prices': [round(p['price_npr'], 2) for p in historical[-7:]] if historical else [],
            }
        except Exception as e:
            logger.error(f"XGBoost prediction failed for {training_name}: {e}, falling back to rule-based")
    
    # Fallback: rule-based prediction
    return _rule_based_predict(original_name, training_name, target_date, historical, live_price)


def _calculate_historical_summary(historical_prices, live_price):
    """
    Calculate historical summary metrics from real data.
    
    Returns:
        dict: Contains last_price, avg_7_days, trend
    """
    if not historical_prices and not live_price:
        return {
            'last_price': 'N/A',
            'avg_7_days': 'N/A',
            'trend': 'stable'
        }
    
    # Get prices list
    prices = []
    if historical_prices:
        prices = [p['price_npr'] for p in historical_prices]
    
    # Last known price (prefer live, then most recent historical)
    if live_price:
        last_price = live_price['avg_price']
    elif prices:
        last_price = prices[-1]
    else:
        last_price = None
    
    # 7-day average
    if len(prices) >= 7:
        avg_7_days = float(np.mean(prices[-7:]))
    elif prices:
        avg_7_days = float(np.mean(prices))
    elif live_price:
        avg_7_days = live_price['avg_price']
    else:
        avg_7_days = None
    
    # Determine trend
    trend = 'stable'
    if len(prices) >= 7:
        recent_avg = np.mean(prices[-3:])
        older_avg = np.mean(prices[-7:-3]) if len(prices) >= 7 else np.mean(prices[:-3])
        
        change_pct = ((recent_avg - older_avg) / older_avg) * 100 if older_avg > 0 else 0
        
        if change_pct > 5:
            trend = 'up'
        elif change_pct < -5:
            trend = 'down'
        else:
            trend = 'stable'
    
    return {
        'last_price': round(last_price, 2) if last_price else 'N/A',
        'avg_7_days': round(avg_7_days, 2) if avg_7_days else 'N/A',
        'trend': trend
    }


def _rule_based_predict(original_name, training_name, target_date, historical, live_price):
    """Fallback rule-based prediction when XGBoost model is unavailable"""
    calendar_features = add_calendar_features(target_date)
    lag_features = calculate_lag_features(historical)
    
    if live_price:
        base_price = live_price['avg_price']
        current_price = live_price['avg_price']
        is_live = True
    else:
        base_price = lag_features['rolling_mean_30']
        current_price = lag_features['price_lag_1']
        is_live = False
    
    seasonal_factor = 1.0
    if calendar_features.get('is_monsoon'):
        seasonal_factor = 1.15
    elif calendar_features.get('is_winter'):
        seasonal_factor = 0.95
    elif calendar_features.get('is_summer'):
        seasonal_factor = 1.05
    
    festival_factor = 1.0
    if calendar_features.get('is_dashain_window'):
        festival_factor = 1.20
    elif calendar_features.get('is_tihar_window'):
        festival_factor = 1.15
    elif calendar_features.get('is_holi_window'):
        festival_factor = 1.08
    
    trend_factor = 1.0 + (lag_features['price_change_7d'] * 0.3)
    trend_factor = max(0.85, min(1.15, trend_factor))
    
    predicted_price = base_price * seasonal_factor * festival_factor * trend_factor
    
    if live_price:
        volatility = (live_price['max_price'] - live_price['min_price']) / 2
    else:
        volatility = lag_features['rolling_std_7']
    
    price_min = max(5, predicted_price - (2 * volatility))
    price_max = predicted_price + (2 * volatility)
    
    recent_prices = [p['price_npr'] for p in historical[-7:]] if historical else []
    
    # Calculate historical summary from real data
    historical_summary = _calculate_historical_summary(historical, live_price)
    
    return {
        'success': True,
        'vegetable': original_name,
        'training_name': training_name,
        'date': target_date.strftime('%Y-%m-%d'),
        'predicted_price': round(predicted_price, 2),
        'price_min': round(price_min, 2),
        'price_max': round(price_max, 2),
        'confidence': 'Medium' if is_live else 'Low',
        'is_live': is_live,
        'model': 'Rule-based (fallback)',
        'historical_summary': historical_summary,
        'factors': {
            'base_price': round(base_price, 2),
            'seasonal_factor': round(seasonal_factor, 2),
            'festival_factor': round(festival_factor, 2),
            'trend_factor': round(trend_factor, 2),
        },
        'recent_prices': [round(p, 2) for p in recent_prices],
    }


def add_calendar_features(target_date):
    """Add calendar-based features for rule-based prediction fallback"""
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    m = target_date.month
    d = target_date.day
    season_map = {1:1, 2:1, 3:2, 4:2, 5:3, 6:4, 7:4, 8:4, 9:5, 10:6, 11:6, 12:1}
    sc = season_map[m]
    return {
        'is_winter': int(sc == 1), 'is_spring': int(sc == 2),
        'is_summer': int(sc == 3), 'is_monsoon': int(sc == 4),
        'is_dashain_window': int(m == 10 and d <= 20),
        'is_tihar_window': int((m == 10 and d >= 20) or (m == 11 and d <= 5)),
        'is_holi_window': int(m == 3 and d <= 15),
    }


def calculate_lag_features(historical_prices):
    """Calculate lag features for rule-based prediction fallback"""
    if not historical_prices or len(historical_prices) == 0:
        return {'price_lag_1': 50.0, 'rolling_mean_30': 50.0, 'rolling_std_7': 5.0, 'price_change_7d': 0.0}
    prices = [p['price_npr'] for p in historical_prices]
    return {
        'price_lag_1': prices[-1],
        'rolling_mean_30': float(np.mean(prices[-30:])) if len(prices) >= 30 else float(np.mean(prices)),
        'rolling_std_7': float(np.std(prices[-7:])) if len(prices) >= 7 else float(np.std(prices)) if len(prices) > 1 else 0,
        'price_change_7d': (prices[-1] - prices[-7]) / prices[-7] if len(prices) >= 7 and prices[-7] != 0 else 0,
    }


def get_price_trends(vegetable, days=30):
    """Get price trends for visualization"""
    # Try to map Kalimati name to training name for historical lookup
    training_name = get_training_vegetable_name(vegetable)
    lookup_name = training_name if training_name else vegetable
    
    historical = get_historical_prices(lookup_name, days=days)

    if not historical:
        return None

    dates = [p['date'].strftime('%Y-%m-%d') if hasattr(p['date'], 'strftime') else str(p['date'])[:10] for p in historical]
    prices = [round(p['price_npr'], 2) for p in historical]

    if len(prices) >= 2:
        change_pct = ((prices[-1] - prices[0]) / prices[0]) * 100
    else:
        change_pct = 0

    return {
        'vegetable': vegetable.title(),
        'training_name': training_name,
        'dates': dates,
        'prices': prices,
        'current_price': prices[-1] if prices else None,
        'avg_price': round(np.mean(prices), 2) if prices else None,
        'min_price': round(min(prices), 2) if prices else None,
        'max_price': round(max(prices), 2) if prices else None,
        'change_pct': round(change_pct, 1),
    }


def get_market_overview():
    """Get overview of all vegetables for dashboard - uses database/cached data"""

    # Try database first (fastest and most reliable)
    try:
        from prices.models import VegetablePrice
        from datetime import date

        latest_prices = VegetablePrice.objects.filter(date=date.today())[:20]

        if latest_prices.exists():
            overview = []
            for item in latest_prices:
                name = item.commodity_name
                base_name = name.split('(')[0].strip()
                training_name = get_training_vegetable_name(name)

                overview.append({
                    'name': name,
                    'local_name': base_name,
                    'price': float(item.avg_price),
                    'min_price': float(item.min_price),
                    'max_price': float(item.max_price),
                    'unit': item.commodity_unit,
                    'change_pct': 0,
                    'trend': 'stable',
                    'is_live': True,
                    'date': str(item.date),
                    'is_predictable': training_name is not None,
                    'training_name': training_name,
                })
            return overview
    except Exception as e:
        logger.error(f"Error getting market overview from database: {e}")
    
    # Fallback to live API (with timeout protection)
    live_service = get_live_data_service()
    
    if USE_LIVE_DATA and live_service:
        try:
            live_prices = live_service.get_all_live_prices()
            if live_prices:
                overview = []
                for item in live_prices[:20]:
                    name = item['name']
                    base_name = name.split('(')[0].strip()
                    training_name = get_training_vegetable_name(name)
                    
                    overview.append({
                        'name': name,
                        'local_name': base_name,
                        'price': item['avg_price'],
                        'min_price': item['min_price'],
                        'max_price': item['max_price'],
                        'unit': item['unit'],
                        'change_pct': 0,
                        'trend': 'stable',
                        'is_live': True,
                        'date': item['date'],
                        'is_predictable': training_name is not None,
                        'training_name': training_name,
                    })
                return overview
        except Exception as e:
            logger.error(f"Error getting live market overview: {e}")
    
    # Final fallback to local historical data
    vegetables = get_vegetables_list()
    overview = []
    
    for veg in vegetables[:12]:
        veg_name = veg['vegetable']
        trends = get_price_trends(veg_name, days=14)
        training_name = get_training_vegetable_name(veg_name)
        
        if trends:
            overview.append({
                'name': veg_name.title(),
                'local_name': veg.get('vegetable_nepali', veg_name.title()),
                'price': trends['current_price'],
                'change_pct': trends['change_pct'],
                'trend': 'up' if trends['change_pct'] > 2 else 'down' if trends['change_pct'] < -2 else 'stable',
                'is_live': False,
                'is_predictable': training_name is not None,
                'training_name': training_name,
            })
    
    return overview


def get_top_movers():
    """Get top price increases and decreases"""
    vegetables = get_vegetables_list()
    all_trends = []
    
    for veg in vegetables:
        trends = get_price_trends(veg['vegetable'], days=7)
        if trends and trends['current_price']:
            all_trends.append({
                'name': veg['vegetable'].title(),
                'local_name': veg.get('vegetable_nepali', ''),
                'price': trends['current_price'],
                'change_pct': trends['change_pct'],
            })
    
    all_trends.sort(key=lambda x: x['change_pct'], reverse=True)
    
    return {
        'top_increases': all_trends[:5],
        'top_decreases': all_trends[-5:][::-1] if len(all_trends) >= 5 else [],
    }
