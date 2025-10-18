"""
Hospital Finder using Google Places API
Finds nearby hospitals based on user location and wound severity
"""
import requests
import os
from math import radians, cos, sin, asin, sqrt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
GOOGLE_PLACES_API_KEY = os.getenv('GOOGLE_PLACES_API_KEY')
# Optional: Use Places API (New) v1 endpoint if set to truthy value
USE_PLACES_API_V1 = os.getenv('GOOGLE_PLACES_API_V1', '').strip().lower() in {'1', 'true', 'yes', 'on'}

# Minimal diagnostics: confirm API key presence without leaking it
if GOOGLE_PLACES_API_KEY:
    try:
        masked = f"{GOOGLE_PLACES_API_KEY[:4]}...{GOOGLE_PLACES_API_KEY[-4:]}"
        print(f"🔑 Google Places API key loaded: {masked}")
    except Exception:
        print("🔑 Google Places API key loaded")
else:
    print("⚠️ GOOGLE_PLACES_API_KEY is missing. Using mock data until it's set.")


def _search_places_v1(lat, lon, is_infected, max_results, radius_km):
    """Use Google Places API (New) v1 to search nearby places.

    Docs: https://developers.google.com/maps/documentation/places/web-service/search-nearby
    Endpoint: POST https://places.googleapis.com/v1/places:searchNearby
    """
    try:
        url = "https://places.googleapis.com/v1/places:searchNearby"
        included_types = ["hospital"] if is_infected else ["doctor"]
        payload = {
            "includedTypes": included_types,
            "maxResultCount": max_results,
            "rankPreference": "DISTANCE",
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lon},
                    # radius in meters
                    "radius": int(radius_km * 1000)
                }
            }
        }
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
            # Only request fields we need to reduce payload/quotas
            "X-Goog-FieldMask": 
                "places.displayName,places.formattedAddress,places.location,"
                "places.rating,places.currentOpeningHours.openNow"
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f"❌ Places v1 HTTP error: {resp.status_code}")
            return None
        data = resp.json()
        if "error" in data:
            print(f"❌ Places v1 error: {data['error'].get('message', 'unknown')}")
            return None
        places = data.get("places", [])
        hospitals = []
        for p in places:
            loc = p.get("location", {})
            place_lat = loc.get("latitude")
            place_lon = loc.get("longitude")
            if place_lat is None or place_lon is None:
                continue
            distance = calculate_distance(lat, lon, place_lat, place_lon)
            hospitals.append({
                "name": (p.get("displayName", {}) or {}).get("text", "Unknown"),
                "address": p.get("formattedAddress", "Address not available"),
                "distance": f"{distance} km",
                "distance_raw": distance,
                "rating": p.get("rating", "N/A"),
                "phone": "N/A",  # Not requested in field mask to save quota
                "lat": place_lat,
                "lon": place_lon,
                "open_now": ((p.get("currentOpeningHours", {}) or {}).get("openNow"))
            })
        hospitals.sort(key=lambda x: x['distance_raw'])
        return hospitals
    except Exception as e:
        print(f"❌ Error in Places v1: {e}")
        return None


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth using Haversine formula
    Returns distance in kilometers
    """
    # Convert to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers
    km = 6371 * c
    return round(km, 1)


def find_nearby_hospitals(lat, lon, is_infected=False, max_results=5):
    """
    Find nearby medical facilities using Google Places API
    
    Args:
        lat (float): Latitude of user location
        lon (float): Longitude of user location  
        is_infected (bool): If True, prioritize emergency rooms. If False, show clinics/urgent care
        max_results (int): Maximum number of results to return (default 5)
        
    Returns:
        list: List of hospital dictionaries with name, address, distance, etc.
    """
    
    if not GOOGLE_PLACES_API_KEY or GOOGLE_PLACES_API_KEY == 'your_google_api_key_here':
        print("⚠️ Google Places API key not configured. Using mock data.")
        return get_mock_hospitals(lat, lon, max_results)
    
    try:
        # Configure search based on infection status
        if is_infected:
            # For infected wounds: prioritize hospitals with emergency services
            search_type = "hospital"
            radius = 25000  # 25km - willing to travel further for emergencies
            keyword = "emergency"
        else:
            # For non-infected: urgent care, clinics are fine
            search_type = "doctor"
            radius = 15000  # 15km - closer is better for minor issues
            keyword = "urgent care"
        
        # Prefer Places API (New) v1 if enabled via env
        if USE_PLACES_API_V1:
            v1_results = _search_places_v1(lat, lon, is_infected, max_results, radius_km=radius/1000 if radius > 1000 else radius/1)
            if v1_results:
                return v1_results[:max_results]
            # If v1 fails (e.g., not enabled), fall back to legacy endpoint

        # Call legacy Google Places API - Nearby Search
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{lat},{lon}",
            "radius": radius,
            "type": search_type,
            "keyword": keyword,
            "key": GOOGLE_PLACES_API_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Google Places API error: {response.status_code}")
            return get_mock_hospitals(lat, lon, max_results)
        
        data = response.json()

        status = data.get('status')
        error_msg = data.get('error_message')
        if status != 'OK':
            if error_msg:
                print(f"❌ Google Places API status: {status} - {error_msg}")
            else:
                print(f"❌ Google Places API status: {status}")
            return get_mock_hospitals(lat, lon, max_results)
        
        # Process results
        hospitals = []
        for place in data.get('results', []):
            place_lat = place['geometry']['location']['lat']
            place_lon = place['geometry']['location']['lng']
            distance = calculate_distance(lat, lon, place_lat, place_lon)
            
            # Get place details if phone number is needed (additional API call)
            # For now, skip to save API quota
            
            hospitals.append({
                "name": place.get('name', 'Unknown'),
                "address": place.get('vicinity', 'Address not available'),
                "distance": f"{distance} km",
                "distance_raw": distance,  # for sorting
                "rating": place.get('rating', 'N/A'),
                "phone": place.get('formatted_phone_number', 'N/A'),
                "lat": place_lat,
                "lon": place_lon,
                "open_now": place.get('opening_hours', {}).get('open_now', None)
            })
        
        # Sort by distance (closest first)
        hospitals.sort(key=lambda x: x['distance_raw'])
        
        # Return top N results
        return hospitals[:max_results]
        
    except Exception as e:
        print(f"❌ Error fetching hospitals: {str(e)}")
        return get_mock_hospitals(lat, lon, max_results)


def get_mock_hospitals(lat, lon, max_results=5):
    """
    Return mock hospital data for testing when API is not available
    Generates hospitals with realistic-looking data based on user's location
    """
    print("ℹ️ Using mock hospital data")
    
    mock_hospitals = [
        {
            "name": "General Hospital",
            "address": "Near your location",
            "distance": f"{calculate_distance(lat, lon, lat + 0.01, lon + 0.01)} km",
            "distance_raw": calculate_distance(lat, lon, lat + 0.01, lon + 0.01),
            "rating": 4.5,
            "phone": "(555) 123-4567",
            "lat": lat + 0.01,
            "lon": lon + 0.01
        },
        {
            "name": "Community Medical Center",
            "address": "Near your location",
            "distance": f"{calculate_distance(lat, lon, lat + 0.02, lon - 0.01)} km",
            "distance_raw": calculate_distance(lat, lon, lat + 0.02, lon - 0.01),
            "rating": 4.2,
            "phone": "(555) 234-5678",
            "lat": lat + 0.02,
            "lon": lon - 0.01
        },
        {
            "name": "Urgent Care Clinic",
            "address": "Near your location",
            "distance": f"{calculate_distance(lat, lon, lat - 0.01, lon + 0.02)} km",
            "distance_raw": calculate_distance(lat, lon, lat - 0.01, lon + 0.02),
            "rating": 4.0,
            "phone": "(555) 345-6789",
            "lat": lat - 0.01,
            "lon": lon + 0.02
        },
        {
            "name": "Regional Medical Center",
            "address": "Near your location",
            "distance": f"{calculate_distance(lat, lon, lat + 0.03, lon - 0.02)} km",
            "distance_raw": calculate_distance(lat, lon, lat + 0.03, lon - 0.02),
            "rating": 4.3,
            "phone": "(555) 456-7890",
            "lat": lat + 0.03,
            "lon": lon - 0.02
        },
        {
            "name": "City Hospital Emergency",
            "address": "Near your location",
            "distance": f"{calculate_distance(lat, lon, lat - 0.02, lon - 0.03)} km",
            "distance_raw": calculate_distance(lat, lon, lat - 0.02, lon - 0.03),
            "rating": 3.9,
            "phone": "(555) 567-8901",
            "lat": lat - 0.02,
            "lon": lon - 0.03
        }
    ]
    
    # Sort by distance
    mock_hospitals.sort(key=lambda x: x['distance_raw'])
    
    return mock_hospitals[:max_results]


if __name__ == "__main__":
    # Test the function
    print("Testing hospital finder...")
    
    # Test with sample coordinates (San Francisco)
    test_lat = 37.7749
    test_lon = -122.4194
    
    print(f"\nSearching for hospitals near ({test_lat}, {test_lon})...")
    
    # Test for infected wound
    print("\n--- Infected Wound (Emergency Priority) ---")
    hospitals = find_nearby_hospitals(test_lat, test_lon, is_infected=True, max_results=5)
    for i, hospital in enumerate(hospitals, 1):
        print(f"{i}. {hospital['name']}")
        print(f"   Address: {hospital['address']}")
        print(f"   Distance: {hospital['distance']}")
        print(f"   Rating: {hospital['rating']}")
        print()
