"""
Auto-sync Middleware

Automatically syncs vegetable prices from Kalimati API once per day
when the application is first accessed each day.
"""
from datetime import date
import threading
import logging

logger = logging.getLogger(__name__)


class AutoSyncMiddleware:
    """
    Middleware to automatically sync daily prices.
    
    Checks if today's data exists in database. If not, triggers
    background sync from Kalimati Market API.
    """
    
    _last_sync_date = None
    _sync_in_progress = False
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Only check on relevant pages
        if request.path.startswith(('/trade/', '/predictions/', '/insights/')):
            self._check_and_sync()
        
        response = self.get_response(request)
        return response
    
    def _check_and_sync(self):
        """Check if sync is needed and trigger if necessary."""
        today = date.today()
        
        # Reset sync flag if it's a new day
        if AutoSyncMiddleware._last_sync_date and AutoSyncMiddleware._last_sync_date < today:
            logger.info(f"New day detected: {today}. Resetting sync status.")
            AutoSyncMiddleware._last_sync_date = None
            AutoSyncMiddleware._sync_in_progress = False
        
        # Skip if already synced today
        if AutoSyncMiddleware._last_sync_date == today:
            return
        
        if AutoSyncMiddleware._sync_in_progress:
            logger.info("Auto-sync already in progress, skipping...")
            return
        
        # Check database for today's data
        try:
            from prices.models import VegetablePrice
            has_today_data = VegetablePrice.objects.filter(date=today).exists()
            
            if not has_today_data:
                logger.info(f"[Auto-sync] No data found for {today}. Starting background sync...")
                AutoSyncMiddleware._sync_in_progress = True
                thread = threading.Thread(target=self._sync_prices)
                thread.daemon = True
                thread.start()
            else:
                # Data exists, mark as synced
                AutoSyncMiddleware._last_sync_date = today
                logger.debug(f"[Auto-sync] Data already exists for {today}. Skipping sync.")
        except Exception as e:
            logger.error(f"[Auto-sync] Check failed: {e}")
    
    def _sync_prices(self):
        """Background task to sync prices."""
        today = date.today()
        try:
            from django.core.management import call_command
            from io import StringIO
            
            logger.info(f"[Auto-sync] Starting sync for {today} (this may take 15-20 seconds)...")
            
            # Capture sync command output
            output = StringIO()
            call_command('sync_prices', timeout=30, stdout=output)
            
            # Mark as synced
            AutoSyncMiddleware._last_sync_date = today
            
            # Log results
            output_text = output.getvalue()
            logger.info(f"[Auto-sync] Completed for {today}")
            
            # Extract summary from output
            if 'created' in output_text:
                import re
                match = re.search(r'(\d+) created', output_text)
                if match:
                    logger.info(f"[Auto-sync] Successfully imported {match.group(1)} new price records")
            
        except Exception as e:
            logger.error(f"[Auto-sync] Failed: {type(e).__name__}: {e}")
        finally:
            AutoSyncMiddleware._sync_in_progress = False
