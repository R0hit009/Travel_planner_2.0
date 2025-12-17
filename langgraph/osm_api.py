import requests

def geocode_place_name(place_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "TouristPlaceFinder/1.0 (your_email2@example.com)"
    }

    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    if not data:
        raise Exception(f"Location not found: {place_name}")
    return float(data[0]['lat']), float(data[0]['lon'])

def get_tourist_places_osm_by_name(place_name, radius=2000, top_n=20):
    lat, lon = geocode_place_name(place_name)
    print(f"{place_name}", lat, lon)
    overpass_url = "http://overpass-api.de/api/interpreter"
    tag_type = 'tourism'
    tags_key = ["aquarium", "artwork", "attraction", "gallery", "museum",
        "picnic_site", "theme_park", "viewpoint", "zoo"]
    tag_filters = " ".join([
        f'node["{tag_type}"="{t}"](around:{radius},{lat},{lon});'
        f'way["{tag_type}"="{t}"](around:{radius},{lat},{lon});'
        f'relation["{tag_type}"="{t}"](around:{radius},{lat},{lon});'
        for t in tags_key
    ])
    query = f"""[out:json];({tag_filters});out center;"""

    response = requests.get(overpass_url, params={'data': query})
    data = response.json()

    places = []
    for element in data['elements'][:top_n]:
        tags = element.get('tags', {})
        name = tags.get('name', 'Unnamed')
        tourism_type = tags.get('tourism', 'Unknown')
        lat = element.get('lat') or element.get('center', {}).get('lat')
        lon = element.get('lon') or element.get('center', {}).get('lon')
        places.append({
            'name': name,
            'type': tourism_type,
            'latitude': lat,
            'longitude': lon
        })

    place_string = "\n".join([f"- {p['name']} ({p['type']}) at {p['latitude']}, {p['longitude']}" for p in places])
    return place_string

# Example usage
if __name__ == "__main__":
    place_name = "digha"
    places = get_tourist_places_osm_by_name(place_name,8000)
    print(places)