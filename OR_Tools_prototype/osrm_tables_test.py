import requests
from typing import List, Tuple

OSRM_TABLE_URL = "https://osrm-caribbean-785077267034.us-central1.run.app/table/v1/driving/"

def build_osrm_time_matrix(locations: List[Tuple[float, float]]) -> List[List[int]]:
    """
    Given a list of (lat, lon) tuples, return the OSRM travel time matrix (in seconds).
    """
    if len(locations) > 100:
        raise ValueError("OSRM table API usually supports up to 100 coordinates")

    # Convert (lat, lon) → OSRM expected format (lon,lat)
    coords = ";".join([f"{lon},{lat}" for lat, lon in locations])
    
    # Build request
    params = {
        "annotations": "duration"
    }
    url = f"{OSRM_TABLE_URL}{coords}"

    print("Requesting OSRM matrix...")
    response = requests.get(url, params=params)
    data = response.json()

    if "durations" not in data:
        raise RuntimeError("OSRM table failed. Check server or coordinates.")
    
    return data["durations"]

# --- Example Usage ---

if __name__ == "__main__":
    # Example coordinates (lat, lon)
    coordinates = [
        (17.1189, -61.8406),  # Agent 1 start
        (17.1500, -61.8500),  # Agent 2 start
        (17.1250, -61.8600),  # Agent 3 start
        (17.1175, -61.8456),  # Pickup 1
        (17.1210, -61.8460),  # Pickup 2
        (17.1240, -61.8450),  # Pickup 3
        (17.1335, -61.8315),  # Delivery 1
        (17.1300, -61.8300),  # Delivery 2
        (17.1400, -61.8290),  # Delivery 3
        (17.1280, -61.8490),  # Pickup 4
        (17.1440, -61.8200),  # Delivery 4
        (17.1260, -61.8470),  # Pickup 5
        (17.1420, -61.8220),  # Delivery 5
    ]

    time_matrix = build_osrm_time_matrix(coordinates)
    print("Time matrix (in seconds):")
    for row in time_matrix:
        print(row)