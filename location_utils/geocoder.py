# location_utils/geocoder.py
import logging
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize geocoder
geolocator = Nominatim(
    user_agent="geoai_app_v2",
    timeout=10
)
reverse_geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)

def get_address_from_coords(coords):
    if not coords or not isinstance(coords, (list, tuple)) or len(coords) != 2:
        logger.info("[GEOCODER] Invalid coordinates input: %s", coords)
        return "Invalid coordinates"

    lat, lon = coords
    
    # Validate coordinate ranges
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        logger.info(f"[GEOCODER] Coordinates out of range: lat={lat}, lon={lon}")
        return "Invalid coordinate range"
    
    for attempt in range(3):  # Retry logic
        try:
            logger.info(f"[GEOCODER] Attempt {attempt + 1}: Reverse geocoding {lat}, {lon}")
            location = reverse_geocode(f"{lat}, {lon}", language='en', exactly_one=True, timeout=15)
            
            if location and location.address:
                logger.info(f"[GEOCODER] Success: {location.address}")
                return location.address
            else:
                logger.info("[GEOCODER] No location found")
                return "Unknown location"
                
        except Exception as e:
            logger.warning(f"[GEOCODER] Attempt {attempt+1} failed: {e}")
            if attempt < 3:  # Don't sleep on last attempt
                time.sleep(2)
                
    logger.error("[GEOCODER] All geocoding attempts failed for %s", coords)
    return "Geocoding service unavailable"
