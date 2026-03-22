"""
Live Data Integration Service

Handles real-time data fetching from Nepal's Kalimati Market API.
Provides caching and error handling for reliable data access.
Uses Selenium with headless Chrome to bypass bot protection.
"""
import cloudscraper
from requests.exceptions import RequestException, Timeout, ConnectionError
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json
import logging

logger = logging.getLogger('veggieezee.live_data_service')

KALIMATI_API_URL = "https://kalimatimarket.gov.np/api/daily-prices/en"

_cached_data = None
_cache_timestamp = None
CACHE_DURATION_MINUTES = 30

scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    }
)

# Import selenium fetcher
try:
    from .selenium_fetcher import fetch_with_selenium
    SELENIUM_AVAILABLE = True
    logger.info("Selenium fetcher available for bot protection bypass")
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.warning("Selenium not available, will use cloudscraper only")


def fetch_live_prices(timeout: int = 3) -> Optional[Dict]:
    """
    Fetch current vegetable prices from Kalimati Market API.
    
    Uses a two-tier approach:
    1. First tries fast cloudscraper (3-5 seconds)
    2. If bot protection detected, falls back to Selenium (15-20 seconds)
    
    Uses in-memory caching to reduce API calls.
    
    Args:
        timeout: Request timeout in seconds for cloudscraper (default: 3)
                 Selenium uses extended timeout (30s) automatically
        
    Returns:
        dict: API response containing prices data, or None if unavailable
    """
    global _cached_data, _cache_timestamp
    
    # Check cache first
    if _cached_data and _cache_timestamp:
        cache_age = datetime.now() - _cache_timestamp
        if cache_age < timedelta(minutes=CACHE_DURATION_MINUTES):
            logger.debug(f"Using cached data (age: {cache_age.seconds}s)")
            return _cached_data
    
    # Try Method 1: Fast cloudscraper
    try:
        logger.info("Attempting fast fetch with cloudscraper...")
        
        response = scraper.get(KALIMATI_API_URL, timeout=timeout)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        
        # Check if we got HTML (challenge page)
        if 'html' in content_type:
            logger.warning("Bot protection challenge detected (HTML response)")
            # Don't return yet, try selenium next
        else:
            # Try to parse JSON
            data = response.json()
            
            if 'message' in data and 'bot-protection' in data.get('message', '').lower():
                logger.warning(f"Bot protection message: {data['message']}")
            elif data.get('status') == 200 and 'prices' in data:
                _cached_data = data
                _cache_timestamp = datetime.now()
                logger.info(f"Cloudscraper success: {len(data['prices'])} prices for {data.get('date')}")
                return data
    
    except json.JSONDecodeError:
        logger.debug("Cloudscraper returned HTML, trying Selenium...")
    except (RequestException, Timeout, ConnectionError) as e:
        logger.debug(f"Cloudscraper failed: {type(e).__name__}")
    except Exception as e:
        logger.error(f"Unexpected error with cloudscraper: {e}")
    
    # Method 2: Selenium (slower but bypasses bot protection)
    if SELENIUM_AVAILABLE:
        try:
            logger.info("Falling back to Selenium (this may take 15-20 seconds)...")
            data = fetch_with_selenium(KALIMATI_API_URL, timeout=30)
            
            if data and data.get('status') == 200 and 'prices' in data:
                _cached_data = data
                _cache_timestamp = datetime.now()
                logger.info(f"Selenium success: {len(data['prices'])} prices for {data.get('date')}")
                return data
            else:
                logger.warning("Selenium fetch returned no valid data")
        except Exception as e:
            logger.error(f"Selenium fetch error: {type(e).__name__}: {e}")
    else:
        logger.warning("Selenium not available, cannot bypass bot protection")
    
    # Final fallback: return cached data if available
    if _cached_data:
        logger.info("Using cached data as fallback")
        return _cached_data
    
    logger.error("All fetch methods failed and no cached data available")
    return None


def get_live_price(commodity_name: str) -> Optional[Dict]:
    """
    Get live price for a specific commodity
    Performs fuzzy matching on commodity names
    """
    data = fetch_live_prices()
    
    if not data or 'prices' not in data:
        return None
    
    search_name = commodity_name.lower().strip()
    
    for item in data['prices']:
        item_name = item['commodityname'].lower()
        
        if search_name in item_name or item_name in search_name:
            return {
                'name': item['commodityname'],
                'unit': item['commodityunit'],
                'min_price': float(item['minprice']),
                'max_price': float(item['maxprice']),
                'avg_price': float(item['avgprice']),
                'date': data.get('date', datetime.now().strftime('%Y-%m-%d')),
                'is_live': True,
            }
    
    return None


def get_all_live_prices() -> List[Dict]:
    """
    Get all live prices from the API
    """
    data = fetch_live_prices()
    
    if not data or 'prices' not in data:
        return []
    
    prices = []
    for item in data['prices']:
        prices.append({
            'name': item['commodityname'],
            'unit': item['commodityunit'],
            'min_price': float(item['minprice']),
            'max_price': float(item['maxprice']),
            'avg_price': float(item['avgprice']),
            'date': data.get('date', datetime.now().strftime('%Y-%m-%d')),
        })
    
    return prices


def get_live_vegetables_list() -> List[Dict]:
    """
    Get list of vegetables from live API with Nepali name mapping
    """
    nepali_names = {
        'tomato': 'Golbheda',
        'potato': 'Aalu',
        'onion': 'Pyaaj',
        'carrot': 'Gajar',
        'cabbage': 'Banda',
        'cauliflower': 'Kauli',
        'cauli': 'Kauli',
        'radish': 'Mula',
        'raddish': 'Mula',
        'brinjal': 'Bhanta',
        'eggplant': 'Bhanta',
        'spinach': 'Palungo',
        'cucumber': 'Kakro',
        'bitter gourd': 'Tite Karela',
        'bottle gourd': 'Lauka',
        'pumpkin': 'Pharsi',
        'beans': 'Simi',
        'peas': 'Kerau',
        'okra': 'Bhindi',
        'ginger': 'Aduwa',
        'garlic': 'Lasun',
        'chilli': 'Khursani',
        'capsicum': 'Bhede Khursani',
        'mushroom': 'Chyau',
        'coriander': 'Dhaniya',
        'mint': 'Pudina',
        'lettuce': 'Salad Patta',
        'broccoli': 'Broccoli',
        'asparagus': 'Kurilo',
        'sweet potato': 'Sakharkhand',
        'yam': 'Tarul',
        'bamboo shoot': 'Tama',
        'drumstick': 'Sahijan',
        'fenugreek': 'Methi',
        'mustard': 'Rayo',
    }
    
    data = fetch_live_prices()
    
    if not data or 'prices' not in data:
        return []
    
    vegetables = []
    seen = set()
    
    for item in data['prices']:
        name = item['commodityname']
        name_lower = name.lower()
        
        nepali = None
        for key, value in nepali_names.items():
            if key in name_lower:
                nepali = value
                break
        
        if nepali is None:
            nepali = name.split('(')[0].strip()
        
        base_name = name.split('(')[0].strip().lower()
        if base_name not in seen:
            seen.add(base_name)
            vegetables.append({
                'vegetable': name,
                'vegetable_nepali': nepali,
                'current_price': float(item['avgprice']),
                'min_price': float(item['minprice']),
                'max_price': float(item['maxprice']),
                'unit': item['commodityunit'],
            })
    
    return vegetables


def search_commodity(query: str) -> List[Dict]:
    """
    Search for commodities matching a query
    """
    data = fetch_live_prices()
    
    if not data or 'prices' not in data:
        return []
    
    query_lower = query.lower().strip()
    results = []
    
    for item in data['prices']:
        if query_lower in item['commodityname'].lower():
            results.append({
                'name': item['commodityname'],
                'unit': item['commodityunit'],
                'min_price': float(item['minprice']),
                'max_price': float(item['maxprice']),
                'avg_price': float(item['avgprice']),
            })
    
    return results


def get_market_date() -> Optional[str]:
    """
    Get the date of the market data
    """
    data = fetch_live_prices()
    
    if data:
        return data.get('date')
    
    return None


def is_api_available() -> bool:
    """
    Check if the Kalimati API is available
    """
    try:
        response = scraper.get(KALIMATI_API_URL, timeout=5)
        return response.status_code == 200
    except Exception:
        return False
