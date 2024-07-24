import json
import folium
import webbrowser
import os

def load_localization_data(json_file):
    """ Load localization data from a line-separated JSON file. """
    data = []
    with open(json_file, 'r') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} in line: {line}")
    return data

def create_map(localizations):
    """Create a folium map with markers for each successful localization and enhanced zoom features."""
    if not localizations:
        print("No data available.")
        return
    
    # Assume the first successful localization's coordinates as the center for the initial map view
    first_loc = next((loc for loc in localizations if loc.get("success") and loc["error"] == "none"), None)
    center = [first_loc['latitude'], first_loc['longitude']] if first_loc else [0, 0]
    map = folium.Map(
        location=center,
        zoom_start=18,  # Start zoom closer
        min_zoom=10,    # Minimum zoom level allowed
        max_zoom=22     # Maximum zoom level allowed
    )

    # Add markers for successful localizations
    for loc in localizations:
        if loc.get('success') and loc['error'] == 'none':
            html = (f'<img src="{loc["image"]}" width="300" height="200"><br>'
                    f'<b>Latitude:</b> {loc["latitude"]}<br>'
                    f'<b>Longitude:</b> {loc["longitude"]}<br>'
                    f'<b>Image Path:</b> {loc["image"]}')
            iframe = folium.IFrame(html=html, width=350, height=250)
            popup = folium.Popup(iframe, max_width=350)
            folium.Marker(
                location=[loc['latitude'], loc['longitude']],
                popup=popup,
                tooltip="Click for details"
            ).add_to(map)

    # Save the map to an HTML file
    map.save('localization_map.html')
    return 'localization_map.html'

def main(json_file):
    data = load_localization_data(json_file)
    html_file = create_map(data)
    # Automatically open the generated map in the default web browser
    if html_file:
        webbrowser.open('file://' + os.path.realpath(html_file), new=2)

if __name__ == "__main__":
    json_file = "localization_results.json"  # Path to your JSON file with localization data
    main(json_file)
