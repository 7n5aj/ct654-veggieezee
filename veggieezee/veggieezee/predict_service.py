"""
Vegetable Price Prediction Service

Core prediction engine that combines historical data with live market prices
to forecast vegetable prices for Nepal's Kalimati Market.
"""
import os
import re
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
ML_CSV_PATH = BASE_DIR / 'predict' / 'ml' / 'processed_vegetable_prices.csv'
ML_KALIMATI_ALIASES_PATH = BASE_DIR / 'predict' / 'ml' / 'kalimati_training_aliases.csv'
ML_DATA_PATH = BASE_DIR / 'predict' / 'ml' / 'data.xlsx'

_model = None
_label_encoder = None
_feature_cols = None
_historical_data = None
_vegetables_list = None

USE_LIVE_DATA = False

_kalimati_norm_to_class = None
_training_class_set = None


def _normalize_commodity_key(name):
    s = str(name).lower().strip()
    s = s.replace('_', ' ')
    s = re.sub(r'[\(\)]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _ensure_kalimati_maps():
    """
    Build Kalimati -> training class map from processed_vegetable_prices.csv (same `name`
    column the notebook trained on) plus kalimati_training_aliases.csv for API strings
    not present as name_original in the main file.
    """
    global _kalimati_norm_to_class, _training_class_set
    if _kalimati_norm_to_class is not None:
        return

    enc_path = MODELS_DIR / 'nepal_veg_label_encoder.pkl'
    if not enc_path.is_file():
        logger.error('Label encoder not found at %s', enc_path)
        _kalimati_norm_to_class = {}
        _training_class_set = frozenset()
        return

    le = joblib.load(enc_path)
    _training_class_set = frozenset(str(c) for c in le.classes_)
    norm_map = {}

    if ML_CSV_PATH.is_file():
        try:
            df = pd.read_csv(ML_CSV_PATH, usecols=['name_original', 'name'])
            pairs = df[['name_original', 'name']].drop_duplicates()
            for _, row in pairs.iterrows():
                cls = str(row['name']).strip()
                if cls not in _training_class_set:
                    continue
                for raw in (row['name_original'], row['name']):
                    k = _normalize_commodity_key(raw)
                    if k and k not in norm_map:
                        norm_map[k] = cls
        except Exception as e:
            logger.error('Could not build Kalimati map from CSV: %s', e)

    if ML_KALIMATI_ALIASES_PATH.is_file():
        try:
            adf = pd.read_csv(ML_KALIMATI_ALIASES_PATH, usecols=['name_original', 'name'])
            for _, arow in adf.drop_duplicates().iterrows():
                cls = str(arow['name']).strip()
                if cls not in _training_class_set:
                    continue
                for raw in (arow['name_original'], arow['name']):
                    k = _normalize_commodity_key(raw)
                    if k:
                        norm_map[k] = cls
        except Exception as e:
            logger.error('Could not load Kalimati aliases CSV: %s', e)

    _kalimati_norm_to_class = norm_map
    logger.info(
        'Kalimati map: %s normalized keys -> %s encoder classes',
        len(_kalimati_norm_to_class),
        len(_training_class_set),
    )


def get_training_vegetables_sorted():
    _ensure_kalimati_maps()
    return sorted(_training_class_set)


def get_training_class_count():
    """Label-encoder classes in the shipped model (full training taxonomy)."""
    _ensure_kalimati_maps()
    return len(_training_class_set) if _training_class_set else 0


def get_training_vegetable_name(kalimati_name):
    """
    Map Kalimati API commodity name to the exact training class string expected by
    nepal_veg_label_encoder.pkl (e.g. 'cauli', 'tomato').
    """
    if not kalimati_name:
        return None
    _ensure_kalimati_maps()
    nk = _normalize_commodity_key(kalimati_name)
    cls = _kalimati_norm_to_class.get(nk)
    if cls is not None and cls in _training_class_set:
        return cls
    kalimati_lower = kalimati_name.strip().lower()
    for c in _training_class_set:
        if c.lower() == kalimati_lower:
            return c
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
    """
    Load historical prices aligned with the XGBoost training data.
    Prefers processed_vegetable_prices.csv (notebook `name` column); falls back to data.xlsx.
    """
    global _historical_data, _vegetables_list

    if _historical_data is not None:
        return _historical_data

    if ML_CSV_PATH.is_file():
        try:
            df = pd.read_csv(ML_CSV_PATH)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date', 'name', 'average'])
            df['vegetable'] = df['name'].astype(str).str.strip().str.lower()
            df['price_npr'] = pd.to_numeric(df['average'], errors='coerce')
            df = df.dropna(subset=['price_npr'])
            if 'name_original' in df.columns:
                df['vegetable_nepali'] = (
                    df['name_original'].astype(str).str.replace('_', ' ', regex=False).str.title()
                )
            else:
                df['vegetable_nepali'] = df['vegetable'].str.replace('_', ' ', regex=False).str.title()
            df['region'] = 'Kalimati'
            df = df.sort_values('date').reset_index(drop=True)
            _historical_data = df
            veg_df = df[['vegetable', 'vegetable_nepali']].drop_duplicates()
            _vegetables_list = veg_df.to_dict('records')
            logger.info('Historical data loaded from CSV (%s rows)', len(df))
            return df
        except Exception as e:
            logger.error('Error loading historical CSV: %s', e)

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


def get_vegetable_nepali_label(commodity_name):
    """
    Display label for Nepali / local name: prefer processed CSV vegetable_nepali
    for the mapped training class, else the commodity name before '('.
    """
    if not commodity_name:
        return ''
    tn = get_training_vegetable_name(commodity_name)
    df = load_historical_data()
    if df is not None and tn:
        col = df.loc[df['vegetable'] == str(tn).lower().strip(), 'vegetable_nepali']
        if len(col) > 0:
            return str(col.iloc[-1])
    base = commodity_name.split('(')[0].strip()
    return base if base else str(commodity_name)


def get_vegetables_list():
    """Get list of available vegetables - prefers database data over live API"""
    global _vegetables_list

    try:
        from django.core.cache import cache
        from django.conf import settings
        from prices.models import VegetablePrice
        from prices.snapshot import effective_price_date

        min_rows = getattr(settings, 'PRICES_SYNC_MIN_ROW_SKIP', 20)
        ttl = getattr(settings, 'PRICES_VEG_LIST_CACHE_TTL', 45)
        price_date = effective_price_date(min_rows)

        if price_date:
            cache_key = f'veg_list:v2:{price_date.isoformat()}'
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

            latest_prices = (
                VegetablePrice.objects.filter(date=price_date)
                .order_by('commodity_name')
                .values(
                    'commodity_name',
                    'avg_price',
                    'min_price',
                    'max_price',
                    'commodity_unit',
                )
            )

            vegetables = []
            for item in latest_prices:
                name = item['commodity_name']
                nepali = get_vegetable_nepali_label(name)
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

            cache.set(cache_key, vegetables, ttl)
            logger.debug(
                'Loaded %s vegetables from DB for %s (%s predictable)',
                len(vegetables),
                price_date,
                sum(1 for v in vegetables if v['is_predictable']),
            )
            return vegetables
    except Exception as e:
        logger.error('Error getting vegetables from database: %s', e)
    
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

    return []


def get_predictable_vegetables():
    """Get only vegetables that can be predicted (exist in training data)"""
    all_vegetables = get_vegetables_list()
    return [v for v in all_vegetables if v.get('is_predictable', False)]


def _format_training_class_label(class_key):
    """Readable title from encoder class (e.g. bitter_gourd → Bitter gourd)."""
    if not class_key:
        return ''
    s = str(class_key).replace('_', ' ').strip()
    if not s:
        return ''
    return s[0].upper() + s[1:] if len(s) > 1 else s.upper()


def _forecast_articulation(use_kalimati_row, live_price):
    """Short user-facing note about data sources for this forecast."""
    if not use_kalimati_row:
        return (
            'Trained commodity not taken from today\'s live market list. '
            'Lags use historical Kalimati and archive data only.'
        )
    if live_price:
        return (
            'Uses today\'s market or synced prices together with historical price lags.'
        )
    return (
        'Lag features come from the synced database and long-run archive; '
        'no separate live API quote was merged for this request.'
    )


def get_prediction_vegetable_groups():
    """
    Split choices for the prediction UI:

    - First group: commodities on the latest Kalimati DB snapshot that map to a class
      (live market names).
    - Second group: remaining encoder classes still predictable from archived CSV history
      even when that item does not appear on today's API list.
    """
    live_catalog = []
    for v in get_predictable_vegetables():
        row = dict(v)
        row['choice_value'] = row['vegetable']
        row['source'] = 'live'
        row['option_hint'] = "On today's Kalimati snapshot"
        live_catalog.append(row)

    covered = {x.get('training_name') for x in live_catalog if x.get('training_name')}
    model_only = []
    for cls in get_training_vegetables_sorted():
        if cls in covered:
            continue
        model_only.append({
            'vegetable': cls,
            'choice_value': cls,
            'training_name': cls,
            'vegetable_nepali': get_vegetable_nepali_label(cls),
            'display_title': _format_training_class_label(cls),
            'current_price': None,
            'min_price': None,
            'max_price': None,
            'unit': 'KG',
            'is_predictable': True,
            'source': 'model_only',
            'option_hint': "Trained class - historical series only; not on today's API list",
        })

    return live_catalog, model_only


def get_all_prediction_choices():
    """Single ordered list for quick-pick sections (live first, then model-only)."""
    a, b = get_prediction_vegetable_groups()
    return a + b


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


def get_historical_prices(vegetable, days=30, region=None, kalimati_commodity=None):
    """
    Get historical prices for a training-class key (e.g. 'cauli', 'tomato').
    Combines database (Kalimati sync) with archived CSV / Excel history.
    """
    all_prices = []

    # STEP 1: Recent prices from database
    try:
        from prices.models import VegetablePrice
        from datetime import timedelta
        from django.utils import timezone
        import pandas as pd

        start_date = timezone.localdate() - timedelta(days=90)

        if kalimati_commodity and str(kalimati_commodity).strip():
            db_qs = VegetablePrice.objects.filter(
                commodity_name__iexact=str(kalimati_commodity).strip(),
                date__gte=start_date,
            ).order_by('date')
        else:
            training_lookup = get_training_vegetable_name(vegetable) or vegetable
            token = (training_lookup or '').replace('_', ' ').split()[0]
            db_qs = VegetablePrice.objects.filter(
                commodity_name__icontains=token,
                date__gte=start_date,
            ).order_by('date')
        db_prices = db_qs.values('date', 'avg_price')
        
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

        logger.debug(
            "Found %s prices from database for %s",
            len(all_prices),
            kalimati_commodity or vegetable,
        )
    except Exception as e:
        logger.error(f"Error fetching from database: {e}")
    
    # STEP 2: Get historical data from Excel file
    df = load_historical_data()
    
    if df is not None:
        vegetable_lower = vegetable.lower().strip()
        mask = df['vegetable'] == vegetable_lower
        
        if region:
            mask &= df['region'] == region
        
        hist_df = df.loc[mask, ['date', 'price_npr']].sort_values('date')
        # Only recent rows are needed for final `days`-length output (DB covers ~90d).
        archive_cap = min(len(hist_df), max(days * 8, 200))
        if archive_cap < len(hist_df):
            hist_df = hist_df.iloc[-archive_cap:]
        if not hist_df.empty:
            dcol = hist_df['date'].values
            pcol = hist_df['price_npr'].values
            for i in range(len(hist_df)):
                try:
                    all_prices.append({
                        'date': pd.Timestamp(dcol[i]),
                        'price_npr': float(pcol[i]),
                        'source': 'archive'
                    })
                except (OSError, ValueError, OverflowError):
                    continue

        logger.debug(
            "Used %s archive rows for %s (of %s available)",
            len(hist_df),
            vegetable_lower,
            int(mask.sum()),
        )

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


def get_xgboost_feature_count():
    return len(XGBOOST_FEATURE_COLS)


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
    # When the form submits the canonical training class (not a Kalimati commodity string),
    # strict DB match on commodity_name would miss — use archive + fuzzy DB path instead.
    use_kalimati_row = (
        original_name and training_name
        and str(original_name).strip().lower() != str(training_name).strip().lower()
    )
    live_price = get_live_price_for_vegetable(original_name) if use_kalimati_row else None
    historical = get_historical_prices(
        vegetable_lower,
        days=60,
        region=region,
        kalimati_commodity=str(original_name).strip() if use_kalimati_row else None,
    )
    forecast_note = _forecast_articulation(use_kalimati_row, live_price)

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
            
            historical_summary = _calculate_historical_summary(
                historical, live_price, predicted_price=predicted_price
            )

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
                    'model_type': f'XGBoost Regressor ({getattr(model, "n_estimators", "?")} trees)',
                    'features_used': len(XGBOOST_FEATURE_COLS),
                    'historical_days': len(historical),
                    'live_data': live_price is not None,
                },
                'recent_prices': [round(p['price_npr'], 2) for p in historical[-7:]] if historical else [],
                'forecast_note': forecast_note,
            }
        except Exception as e:
            logger.error(f"XGBoost prediction failed for {training_name}: {e}, falling back to rule-based")
    
    # Fallback: rule-based prediction
    return _rule_based_predict(
        original_name, training_name, target_date, historical, live_price,
        forecast_note=forecast_note,
    )


def _calculate_historical_summary(historical_prices, live_price, predicted_price=None):
    """
    Summary for the prediction UI: last print, 7d average, trend.

    Trend rules (prediction page):
    1) If a forecast exists, compare predicted vs last known price (>2% / <-2%) so the
       badge matches direction toward the forecast (e.g. 425 -> 400 => down).
    2) If that comparison is roughly flat, use momentum over the latest up-to-7 points
       (same idea as the past-7-days chart): first vs last in that window, >3% / <-3%.
    Live/api price replaces the final historical point so last/avg align with the card.
    """
    if not historical_prices and not live_price:
        return {
            'last_price': 'N/A',
            'avg_7_days': 'N/A',
            'trend': 'stable',
        }

    prices_work = []
    if historical_prices:
        prices_work = [float(p['price_npr']) for p in historical_prices]
    if live_price:
        lp = float(live_price['avg_price'])
        if prices_work:
            prices_work[-1] = lp
        else:
            prices_work = [lp]

    last_price = prices_work[-1] if prices_work else None
    if last_price is None:
        return {
            'last_price': 'N/A',
            'avg_7_days': 'N/A',
            'trend': 'stable',
        }

    if len(prices_work) >= 7:
        avg_7_days = float(np.mean(prices_work[-7:]))
    else:
        avg_7_days = float(np.mean(prices_work))

    window = prices_work[-7:] if len(prices_work) >= 7 else list(prices_work)
    trend_hist = 'stable'
    if len(window) >= 2 and window[0] > 0:
        ch = (window[-1] - window[0]) / window[0] * 100
        if ch > 3:
            trend_hist = 'up'
        elif ch < -3:
            trend_hist = 'down'

    trend = trend_hist
    if predicted_price is not None and last_price and float(last_price) > 0:
        fp = (float(predicted_price) - float(last_price)) / float(last_price) * 100
        if fp >= 2:
            trend = 'up'
        elif fp <= -2:
            trend = 'down'
        else:
            trend = trend_hist

    return {
        'last_price': round(float(last_price), 2),
        'avg_7_days': round(avg_7_days, 2) if avg_7_days is not None else 'N/A',
        'trend': trend,
    }


def _rule_based_predict(
    original_name,
    training_name,
    target_date,
    historical,
    live_price,
    *,
    forecast_note=None,
):
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
    
    historical_summary = _calculate_historical_summary(
        historical, live_price, predicted_price=predicted_price
    )

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
        'forecast_note': forecast_note
        or _forecast_articulation(
            original_name.strip().lower() != (training_name or '').strip().lower(),
            live_price,
        ),
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

    historical = get_historical_prices(
        lookup_name,
        days=days,
        kalimati_commodity=vegetable,
    )

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


def _sparkline_and_trend_from_name(commodity_name, card_price):
    """
    Last ~7 observed prices for mini-charts; last value pinned to card_price so the
    sparkline matches the big Rs. figure. Trend/changes follow that series.
    """
    trends = get_price_trends(commodity_name, days=7)
    price = float(card_price)
    if not trends or not trends.get('prices'):
        return [price, price], 0.0, 'stable'

    spark = [float(p) for p in trends['prices']]
    if not spark:
        return [price, price], 0.0, 'stable'
    if spark[-1] != price:
        spark = list(spark)
        spark[-1] = price
    if len(spark) == 1:
        spark = [price, price]
    if len(spark) >= 2:
        a, b = spark[0], spark[-1]
        ch = ((b - a) / a) * 100 if a else 0.0
        tr = 'up' if ch > 2 else 'down' if ch < -2 else 'stable'
        return spark, round(float(ch), 1), tr
    return [price, price], 0.0, 'stable'


def get_market_overview():
    """Get overview of all vegetables for dashboard - uses database/cached data"""

    # Try database first (fastest and most reliable)
    try:
        from django.conf import settings
        from prices.models import VegetablePrice
        from prices.snapshot import effective_price_date

        min_rows = getattr(settings, 'PRICES_SYNC_MIN_ROW_SKIP', 20)
        price_date = effective_price_date(min_rows)

        latest_prices = (
            VegetablePrice.objects.filter(date=price_date)
            .order_by('commodity_name')[:20]
            if price_date
            else VegetablePrice.objects.none()
        )

        if latest_prices.exists():
            overview = []
            for item in latest_prices:
                name = item.commodity_name
                training_name = get_training_vegetable_name(name)
                avg = float(item.avg_price)
                sparkline, change_pct, trend = _sparkline_and_trend_from_name(name, avg)

                overview.append({
                    'name': name,
                    'local_name': get_vegetable_nepali_label(name),
                    'price': avg,
                    'min_price': float(item.min_price),
                    'max_price': float(item.max_price),
                    'unit': item.commodity_unit,
                    'change_pct': change_pct,
                    'trend': trend,
                    'is_live': True,
                    'date': str(item.date),
                    'is_predictable': training_name is not None,
                    'training_name': training_name,
                    'sparkline_prices': sparkline,
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
                    avg = float(item['avg_price'])
                    sparkline, change_pct, trend = _sparkline_and_trend_from_name(name, avg)
                    overview.append({
                        'name': name,
                        'local_name': base_name,
                        'price': avg,
                        'min_price': float(item['min_price']),
                        'max_price': float(item['max_price']),
                        'unit': item['unit'],
                        'change_pct': change_pct,
                        'trend': trend,
                        'is_live': True,
                        'date': item['date'],
                        'is_predictable': training_name is not None,
                        'training_name': training_name,
                        'sparkline_prices': sparkline,
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
            cp = float(trends['current_price']) if trends['current_price'] is not None else 0.0
            sp = [float(p) for p in trends['prices']] if trends.get('prices') else [cp, cp]
            if sp:
                sp = list(sp)
                sp[-1] = cp
            overview.append({
                'name': veg_name.title(),
                'local_name': veg.get('vegetable_nepali', veg_name.title()),
                'price': cp,
                'change_pct': trends['change_pct'],
                'trend': 'up' if trends['change_pct'] > 2 else 'down' if trends['change_pct'] < -2 else 'stable',
                'is_live': False,
                'is_predictable': training_name is not None,
                'training_name': training_name,
                'sparkline_prices': sp if len(sp) > 1 else [cp, cp],
            })
    
    return overview


def get_top_movers():
    """
    Top price increases/decreases over ~7 days using recent DB rows only.
    Avoids N calls to get_historical_prices (which scans the full archive CSV per item).
    """
    try:
        from prices.models import VegetablePrice
        from datetime import timedelta
        from collections import defaultdict
        from django.utils import timezone

        today = timezone.localdate()
        start = today - timedelta(days=14)
        rows = VegetablePrice.objects.filter(
            date__gte=start,
            date__lte=today,
        ).order_by('commodity_name', 'date').values(
            'commodity_name', 'date', 'avg_price'
        )

        series = defaultdict(list)
        for r in rows:
            series[r['commodity_name']].append(
                (r['date'], float(r['avg_price']))
            )

        all_trends = []
        nepali_by_commodity = {
            name: get_vegetable_nepali_label(name) for name in series
        }

        for name, pts in series.items():
            if len(pts) < 2:
                continue
            first_price = pts[0][1]
            last_price = pts[-1][1]
            if first_price <= 0:
                continue
            change_pct = ((last_price - first_price) / first_price) * 100
            all_trends.append({
                'name': name,
                'local_name': nepali_by_commodity.get(name, ''),
                'price': last_price,
                'change_pct': round(change_pct, 1),
            })

        all_trends.sort(key=lambda x: x['change_pct'], reverse=True)

        return {
            'top_increases': all_trends[:5],
            'top_decreases': all_trends[-5:][::-1] if len(all_trends) >= 5 else [],
        }
    except Exception as e:
        logger.error('get_top_movers failed: %s', e)
        return {'top_increases': [], 'top_decreases': []}
