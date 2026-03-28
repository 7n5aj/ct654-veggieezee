"""
Auto-sync Middleware

Ensures Kalimati daily prices are loaded into the database when users open the site.
Skips work if today's snapshot already exists (see sync_prices command).
"""
import threading
import logging

logger = logging.getLogger(__name__)

_SKIP_PATH_PREFIXES = (
    '/static/',
    '/media/',
)


def _should_run_sync(request):
    if request.method != 'GET':
        return False
    path = request.path or '/'
    if path.startswith(_SKIP_PATH_PREFIXES):
        return False
    # Health checks / tooling
    if path in ('/favicon.ico', '/robots.txt'):
        return False
    return True


class AutoSyncMiddleware:
    """
    On normal page GETs, ensure DB has been hydrated from Kalimati for the current day.

    Heavy lifting (HTTP to Kalimati) runs in a background thread so page responses stay fast.
    Actual API calls are skipped by sync_prices when rows for today already exist.
    """

    _last_sync_date = None
    _sync_in_progress = False

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if _should_run_sync(request):
            self._check_and_sync()

        response = self.get_response(request)
        return response

    def _check_and_sync(self):
        """Check if sync is needed and trigger if necessary."""
        from django.utils import timezone

        local_today = timezone.localdate()

        if AutoSyncMiddleware._last_sync_date and AutoSyncMiddleware._last_sync_date < local_today:
            logger.info('New day detected: %s. Resetting sync status.', local_today)
            AutoSyncMiddleware._last_sync_date = None
            AutoSyncMiddleware._sync_in_progress = False

        if AutoSyncMiddleware._last_sync_date == local_today:
            return

        if AutoSyncMiddleware._sync_in_progress:
            logger.debug('Auto-sync already in progress, skipping...')
            return

        try:
            from django.conf import settings
            from django.core.cache import cache
            from prices.snapshot import should_skip_api_fetch

            min_rows = getattr(settings, 'PRICES_SYNC_MIN_ROW_SKIP', 20)
            day_done_key = f'price_sync_done:{local_today.isoformat()}'

            if cache.get(day_done_key):
                AutoSyncMiddleware._last_sync_date = local_today
                logger.debug(
                    '[Auto-sync] Sync already ran once for %s (cache). Skipping thread.',
                    local_today,
                )
                return

            if should_skip_api_fetch(min_rows):
                AutoSyncMiddleware._last_sync_date = local_today
                logger.debug(
                    '[Auto-sync] DB has %s+ rows for %s. Skipping thread.',
                    min_rows,
                    local_today,
                )
                return

            today_count = VegetablePrice.objects.filter(date=local_today).count()
            logger.info(
                '[Auto-sync] Need snapshot for %s (today rows=%s). Starting background sync...',
                local_today,
                today_count,
            )
            AutoSyncMiddleware._sync_in_progress = True
            thread = threading.Thread(target=self._sync_prices)
            thread.daemon = True
            thread.start()
        except Exception as e:
            logger.error('[Auto-sync] Check failed: %s', e)

    def _sync_prices(self):
        try:
            from django.core.management import call_command
            from django.conf import settings
            from django.utils import timezone
            from io import StringIO
            from prices.snapshot import should_skip_api_fetch

            local_today = timezone.localdate()
            logger.info('[Auto-sync] Running sync_prices for %s ...', local_today)

            output = StringIO()
            call_command('sync_prices', timeout=30, stdout=output)

            min_rows = getattr(settings, 'PRICES_SYNC_MIN_ROW_SKIP', 20)
            day_done_key = f'price_sync_done:{local_today.isoformat()}'
            from django.core.cache import cache

            if cache.get(day_done_key) or should_skip_api_fetch(min_rows):
                AutoSyncMiddleware._last_sync_date = local_today

            output_text = output.getvalue().strip()
            if output_text:
                logger.info('[Auto-sync] Completed: %s', output_text[-200:])

        except Exception as e:
            logger.error('[Auto-sync] Failed: %s: %s', type(e).__name__, e)
        finally:
            AutoSyncMiddleware._sync_in_progress = False
