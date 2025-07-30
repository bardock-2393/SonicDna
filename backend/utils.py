import time
from functools import wraps

def retry_on_failure(max_retries=3, delay=1):
    """Retry decorator for API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed, retrying in {delay} seconds: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator 