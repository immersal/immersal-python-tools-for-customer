from typing import List
import os
import concurrent.futures
from natsort import natsorted # pip install natsort
import struct
import numpy as np # pip install numpy
import json
import requests # pip install requests

def s2i(s: str) -> int:
    try:
        f = s.zfill(8)
        return struct.unpack('>i', bytes.fromhex(f))[0]
    except:
        return 0

def s2f(s: str) -> float:
    try:
        return struct.unpack('>f', bytes.fromhex(s))[0]
    except:
        return 0

def s2d(s: str) -> float:
    try:
        return struct.unpack('>d', bytes.fromhex(s))[0]
    except:
        return 0

def ParseFilenameToImageData(filename: str) -> dict:
    components = os.path.splitext(os.path.basename(filename))[0].split('_')
    imageData = {
        "run": s2i(components[1]),
        "index": int(components[2], 16),
        "px": s2f(components[3]),
        "py": s2f(components[4]),
        "pz": s2f(components[5]),
        "fx": s2f(components[6]),
        "fy": s2f(components[7]),
        "ox": s2f(components[8]),
        "oy": s2f(components[9]),
        "latitude": s2d(components[13]),
        "longitude": s2d(components[14]),
        "altitude": s2d(components[15])
    }
    return imageData

def LocalizeImage(url: str, token: str, mapId: int, imagesList: List[str], index: int, results_file: str) -> None:
    complete_url = url + "/geopose"
    with open(imagesList[index], 'rb') as imgFile:
        img_bytes = imgFile.read()
        imageData = ParseFilenameToImageData(imagesList[index])
        data = {
            "token": token,
            "mapIds": [{"id": mapId}],
            "fx": imageData['fx'],
            "fy": imageData['fy'],
            "ox": imageData['ox'],
            "oy": imageData['oy']
        }
        json_data = json.dumps(data)
        json_bytes = json_data.encode()
        body = json_bytes + b"\0" + img_bytes
        r = requests.post(complete_url, data=body)
        result = json.loads(r.text)
        result['image'] = imagesList[index]  # Add image name to the result for reference

        # Print and write to file
        print(f"Localized {imagesList[index]}: {result}")
        with open(results_file, 'a') as f:
            json.dump(result, f)
            f.write('\n')  # For readability in the JSON file

def main(url: str, token: str, inputDirectory: str, mapId: int, maxThreads: int=1) -> None:
    imagesList = [os.path.join(inputDirectory, file) for file in natsorted(os.listdir(inputDirectory)) if file.endswith('.png')]
    results_file = 'localization_results.json'
    with open(results_file, 'w') as f:  # Create or clear the results file
        f.write('')  # Start with an empty file

    with concurrent.futures.ThreadPoolExecutor(max_workers=maxThreads) as executor:
        futures = [executor.submit(LocalizeImage, url, token, mapId, imagesList, i, results_file) for i in range(len(imagesList))]

    print(f'Finished localizing images. Results written to {results_file}.')

if __name__ == '__main__':
    url = 'https://api.immersal.com'
    token = 'your-developer-token-here'
    inputDirectory = 'path-to-images-here'
    mapId = 00000

    main(url, token, inputDirectory, mapId)
