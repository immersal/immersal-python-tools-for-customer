# Immersal Map Construction for NavLive

This script is designed to process data from NavLive laser scanner and convert it into Immersal VPS maps. It handles the conversion of coordinate systems, image processing, and map construction through the Immersal API.

## Prerequisites

Install the required Python packages:
```bash
pip install requests natsort numpy open3d opencv-python
```

## Usage

1. Modify the following parameters in the `main()` function:
   - `url`: Your Immersal server URL (default: 'https://api.immersal.com/')
   - `directory`: Path to your NavLive data directory
   - `token`: Your Immersal API token
   - `map_name`: Name for your map (must be without special characters and space)

2. Run the script:
```bash
python mapping/navlive/submit_navlive_json.py
```