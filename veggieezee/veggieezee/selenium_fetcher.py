"""
Selenium-based API fetcher to bypass bot protection.

Uses a headless Chrome browser to solve JavaScript challenges
and fetch data from Kalimati Market API.
"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import json
import logging
import time

logger = logging.getLogger('veggieezee.selenium_fetcher')


def fetch_with_selenium(url: str, timeout: int = 30):
    """
    Fetch API data using Selenium with headless Chrome.
    
    Args:
        url: API endpoint URL
        timeout: Maximum wait time in seconds
        
    Returns:
        dict: Parsed JSON response, or None if failed
    """
    driver = None
    try:
        logger.info("Initializing headless Chrome browser...")
        
        # Configure Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless=new')  # New headless mode
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Set user agent
        chrome_options.add_argument(
            'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        )
        
        # Initialize driver with webdriver-manager
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Set page load timeout
        driver.set_page_load_timeout(timeout)
        
        logger.info(f"Loading URL: {url}")
        driver.get(url)
        
        # Wait for page to complete any JS challenges
        # Look for either JSON response or challenge page
        max_wait = timeout
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            page_text = driver.find_element(By.TAG_NAME, 'body').text
            
            # Check if we got JSON
            if page_text.strip().startswith('{'):
                try:
                    data = json.loads(page_text)
                    logger.info("Successfully retrieved JSON data")
                    return data
                except json.JSONDecodeError:
                    pass
            
            # Check for challenge page
            if 'One moment' in page_text or 'please wait' in page_text.lower():
                logger.debug("Bot challenge detected, waiting...")
                time.sleep(2)
            else:
                # Unknown page content
                break
            
            time.sleep(1)
        
        # Final attempt to parse page content
        page_text = driver.find_element(By.TAG_NAME, 'body').text
        if page_text.strip().startswith('{'):
            data = json.loads(page_text)
            logger.info("Successfully retrieved JSON data after waiting")
            return data
        
        logger.warning(f"Page did not return JSON. Content: {page_text[:200]}")
        return None
        
    except Exception as e:
        logger.error(f"Selenium fetch error: {type(e).__name__}: {e}")
        return None
        
    finally:
        if driver:
            try:
                driver.quit()
                logger.debug("Browser closed")
            except:
                pass


def test_selenium_fetch():
    """Test function for debugging"""
    url = "https://kalimatimarket.gov.np/api/daily-prices/en"
    print("Testing Selenium fetcher...")
    print("="*60)
    
    data = fetch_with_selenium(url, timeout=30)
    
    if data:
        print(f"[SUCCESS!]")
        print(f"Status: {data.get('status')}")
        print(f"Date: {data.get('date')}")
        if 'prices' in data:
            print(f"Prices: {len(data['prices'])} items")
            if len(data['prices']) > 0:
                print(f"\nFirst 3 items:")
                for item in data['prices'][:3]:
                    print(f"  - {item['commodityname']}: Rs. {item['avgprice']}")
        else:
            print(f"Response keys: {list(data.keys())}")
            if 'message' in data:
                print(f"Message: {data['message']}")
    else:
        print("[FAILED] - No data retrieved")
    
    print("="*60)


if __name__ == '__main__':
    # Configure logging for standalone test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_selenium_fetch()
