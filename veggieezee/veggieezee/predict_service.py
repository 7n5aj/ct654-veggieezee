"""
Vegetable Price Prediction Service

Core prediction engine that combines historical data with live market prices
to forecast vegetable prices for Nepal's Kalimati Market.
"""
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path

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
        print(f"Error loading historical data: {e}")
        return None


def get_vegetables_list():
    """Get list of available vegetables - prefers database data over live API"""
    global _vegetables_list
    
    # Try to get from database first (faster and more reliable)
    try:
        from prices.models import VegetablePrice
        from datetime import date
        
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
            
            return vegetables
    except Exception as e:
        print(f"Error getting vegetables from database: {e}")
    
    # Fallback to live API (with short timeout)
    if USE_LIVE_DATA:
        live_service = get_live_data_service()
        if live_service:
            try:
                live_vegs = live_service.get_live_vegetables_list()
                if live_vegs:
                    return live_vegs
            except Exception as e:
                print(f"Error getting live vegetables: {e}")
    
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
        print(f"Error getting live price: {e}")
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
            all_prices.append({
                'date': pd.Timestamp(item['date']),
                'price_npr': float(item['avg_price']),
                'source': 'database'
            })
        
        print(f"Found {len(all_prices)} prices from database for {search_name}")
    except Exception as e:
        print(f"Error fetching from database: {e}")
    
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
            all_prices.append({
                'date': pd.Timestamp(row['date']),
                'price_npr': float(row['price_npr']),
                'source': 'excel'
            })
        
        print(f"Found {len(hist_df)} prices from Excel for {vegetable_lower}")
    
    # STEP 3: Combine and deduplicate (database takes precedence over Excel for same dates)
    if not all_prices:
        return []
    
    # Sort by date
    all_prices.sort(key=lambda x: x['date'])
    
    # Remove duplicates - keep database price if both exist for same date
    seen_dates = {}
    for price in all_prices:
        date_key = price['date'].strftime('%Y-%m-%d')
        
        # Database prices take precedence
        if date_key not in seen_dates or price['source'] == 'database':
            seen_dates[date_key] = {
                'date': price['date'],
                'price_npr': price['price_npr']
            }
    
    # Convert back to list and get last N days
    combined_prices = list(seen_dates.values())
    combined_prices.sort(key=lambda x: x['date'])
    
    # Return only the requested number of days (most recent)
    result = combined_prices[-days:] if len(combined_prices) > days else combined_prices
    
    print(f"Returning {len(result)} combined prices (requested {days} days)")
    return result


def add_calendar_features(target_date):
    """Add calendar-based features for prediction"""
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    m = target_date.month
    d = target_date.day
    
    features = {
        'month': m,
        'day': d,
        'year': target_date.year,
        'dayofweek': target_date.dayofweek,
        'weekofyear': target_date.isocalendar()[1],
        'dayofyear': target_date.timetuple().tm_yday,
    }
    
    season_map = {1:1, 2:1, 3:2, 4:2, 5:3, 6:4, 7:4, 8:4, 9:5, 10:6, 11:6, 12:1}
    features['season_code'] = season_map[m]
    features['is_winter'] = int(features['season_code'] == 1)
    features['is_spring'] = int(features['season_code'] == 2)
    features['is_summer'] = int(features['season_code'] == 3)
    features['is_monsoon'] = int(features['season_code'] == 4)
    features['is_autumn'] = int(features['season_code'] == 5)
    features['is_prewin'] = int(features['season_code'] == 6)
    
    monsoon_map = {1:0, 2:0, 3:0, 4:0, 5:0.5, 6:2, 7:3, 8:2.5, 9:1, 10:0, 11:0, 12:0}
    features['monsoon_intensity'] = monsoon_map[m]
    
    features['is_dashain_window'] = int(m == 10 and d <= 20)
    features['is_tihar_window'] = int((m == 10 and d >= 20) or (m == 11 and d <= 5))
    features['is_holi_window'] = int(m == 3 and d <= 15)
    features['is_any_festival'] = int(
        features['is_dashain_window'] or 
        features['is_tihar_window'] or 
        features['is_holi_window']
    )
    
    features['month_sin'] = np.sin(2 * np.pi * m / 12)
    features['month_cos'] = np.cos(2 * np.pi * m / 12)
    
    return features


def calculate_lag_features(historical_prices):
    """Calculate lag and rolling features from historical prices"""
    if not historical_prices or len(historical_prices) == 0:
        return {
            'price_lag_1': 50.0,
            'price_lag_7': 50.0,
            'rolling_mean_7': 50.0,
            'rolling_mean_30': 50.0,
            'rolling_std_7': 5.0,
            'price_change_7d': 0.0,
        }
    
    prices = [p['price_npr'] for p in historical_prices]
    
    features = {
        'price_lag_1': prices[-1] if len(prices) >= 1 else np.mean(prices),
        'price_lag_7': prices[-7] if len(prices) >= 7 else prices[0],
        'rolling_mean_7': np.mean(prices[-7:]) if len(prices) >= 7 else np.mean(prices),
        'rolling_mean_30': np.mean(prices[-30:]) if len(prices) >= 30 else np.mean(prices),
        'rolling_std_7': np.std(prices[-7:]) if len(prices) >= 7 else np.std(prices) if len(prices) > 1 else 0,
        'price_change_7d': (prices[-1] - prices[-7]) / prices[-7] if len(prices) >= 7 and prices[-7] != 0 else 0,
    }
    
    return features


def predict_price(vegetable, target_date, region=None):
    """
    Predict vegetable price for a future date
    Uses live data + historical trends and seasonal patterns
    Maps Kalimati API names to training dataset names for prediction
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    # Get the training dataset vegetable name
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
    
    # Use training name for historical data lookup
    vegetable_lower = training_name.lower().strip()
    
    # Get live price using original Kalimati name
    live_price = get_live_price_for_vegetable(original_name)
    
    # Get historical data using training name
    historical = get_historical_prices(vegetable_lower, days=60, region=region)
    
    if not historical and not live_price:
        return {
            'success': False,
            'error': f'No historical data found for {training_name}',
            'vegetable': original_name,
            'training_name': training_name,
            'date': target_date.strftime('%Y-%m-%d'),
        }
    
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
    if calendar_features['is_monsoon']:
        seasonal_factor = 1.15
    elif calendar_features['is_winter']:
        seasonal_factor = 0.95
    elif calendar_features['is_summer']:
        seasonal_factor = 1.05
    
    festival_factor = 1.0
    if calendar_features['is_dashain_window']:
        festival_factor = 1.20
    elif calendar_features['is_tihar_window']:
        festival_factor = 1.15
    elif calendar_features['is_holi_window']:
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
    
    return {
        'success': True,
        'vegetable': original_name,
        'training_name': training_name,
        'date': target_date.strftime('%Y-%m-%d'),
        'predicted_price': round(predicted_price, 2),
        'price_min': round(price_min, 2),
        'price_max': round(price_max, 2),
        'confidence': 'High' if is_live else 'Medium',
        'is_live': is_live,
        'factors': {
            'base_price': round(base_price, 2),
            'seasonal_factor': round(seasonal_factor, 2),
            'festival_factor': round(festival_factor, 2),
            'trend_factor': round(trend_factor, 2),
        },
        'historical_summary': {
            'last_price': round(current_price, 2) if current_price else None,
            'avg_7_days': round(np.mean(recent_prices), 2) if recent_prices else round(base_price, 2),
            'trend': 'up' if lag_features['price_change_7d'] > 0.02 else 'down' if lag_features['price_change_7d'] < -0.02 else 'stable',
        },
        'live_data': {
            'current_price': live_price['avg_price'] if live_price else None,
            'min_price': live_price['min_price'] if live_price else None,
            'max_price': live_price['max_price'] if live_price else None,
            'market_date': live_price['date'] if live_price else None,
        } if live_price else None
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
        print(f"Error getting market overview from database: {e}")
    
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
            print(f"Error getting live market overview: {e}")
    
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
