import json
import requests # pip install requests
import os

def UploadChunked(url: str, b2gFilepath: str, token: str, chunk_size:int) -> str:
    complete_url = url + '/upload'

    # Initialize the hash and offset
    file_hash = None
    offset = 0
    file_size = os.path.getsize(b2gFilepath)

    # Read the file in binary mode
    with open(b2gFilepath, 'rb') as file:
        while True:
            # Read a chunk of data
            data = file.read(chunk_size)
            
            if not data:
                break

            payload = {
            'token': token,
            'offset': offset
            }

            # Add the hash to the payload
            if file_hash is not None:
                payload['hash'] = file_hash

            json_data = json.dumps(payload)
            json_bytes = json_data.encode()

            body = json_bytes + b"\0" + data

            # Send the chunk to the API endpoint
            r = requests.post(complete_url, data=body)

            # Check response status and error
            response_json = r.json()
            if r.status_code != 200 or response_json.get('error') != 'none':
                print(f"Failed to send data.Error: {response_json.get('error')}")
                break

            # If this is the first chunk, get the hash
            if file_hash is None:
                file_hash = response_json.get('hash')

            # Update the offset for the next chunk
            offset += chunk_size

            print(f'Uploaded {round((offset/file_size)*100, 1)}%')

    return file_hash

def SubmitB2G(b2gFilepath: str, token: str, chunk_size: int) -> str:
    """Returns any errors or the path of the submitted .b2g file in the Cloud Service as a JSON formatted string. Submits an image to the user's workspace."""

    complete_url = url + "/b2g"

    file_hash = UploadChunked(url, b2gFilepath, token, chunk_size)

    if file_hash is None:
        print('Error uploading chunked b2g')
        return 'error'

    # Notice the rotation matrix for the .b2g file
    data = {
        "token": token,
        "hash" : file_hash,
        "px": 0.0,
        "py": 0.0,
        "pz": 0.0,
        "r00": 1.0,
        "r01": 0.0,
        "r02": 0.0,
        "r10": 0.0,
        "r11": 0.0,
        "r12": 1.0,
        "r20": 0.0,
        "r21": -1.0,
        "r22": 0.0,
    }

    json_data = json.dumps(data)
    json_bytes = json_data.encode()

    body = json_bytes

    r = requests.post(complete_url, data=body)
    print(r.text)
    return r.text

def ClearWorkspace(url: str, token: str, deleteAnchor: bool=True) -> str:
    """Returns any errors for the request. Clears the user's workspace.

    If anchor is set to False, the anchor image is kept in the workspace if one exists.
    When set to True, the whole workspace is cleared
    """

    complete_url = url + "/clear"

    data = {
        "token": token,
        "anchor": deleteAnchor,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    print(r.text)
    return r.text


def StartMapConstruction(url: str, token: str, mapName: str) -> str:
    
    """Returns any errors or the new map's id and size (number of images) on the Cloud Service as a JSON formatted string. Submits the map construction job for the images in the user's workspace.
    preservePoses               Boolean to enable constraints input images' camera pose data as constraints. Speeds up map construction if input pose data is accurate
    featureCount                Integer for the max amount of features per image. Increases the total amount of features in the map but also increases map filesize
    featureCountMax             Maximum num of features extracted from image
    featureFilter               Possible values 0 or 1, in scenes where there’s lots of unique details like grass, bushes, leaves, gravel etc we’d pick a lot of noisy details (high frequency features) as our top picks. These features were very hard to localize against. By using this parameter in map construction, it sorts the detected features based on size favoring large features (low frequency features). This seems to improve map construction and localization rate in high frequency environments.
    trackLengthMin              This value represents the minimum number of images from which a feature point must originate to be kept in the map. The larger the number, the higher the reliability required for feature points, resulting in fewer points being retained and a smaller map. We usually start with a default value of 2 and gradually increase it (typically between 2 and 5), observing the success rate of positioning until it noticeably drops. At this point, the map has a sufficiently good positioning success rate while being as small as possible.
    triangulationDistanceMax    This value represents the maximum distance to the target object for positioning, measured in meters. A value of 512 means that the system supports constructing a point cloud for an object up to 512 meters away. It's important to note that in triangulation, if your target object is far away, your baseline (the distance moved laterally) should be correspondingly increased, typically to 5%-10% of the distance. Otherwise, the point clouds for these distant objects will be unreliable. It is also important to note that sometimes constructing point clouds for distant objects is not a good idea because the accuracy of positioning decreases with distance. Therefore, you should adjust this parameter based on the actual scenario.
    dense                       By setting this parameter to 0, it will skip the dense map and glb file generation, which can make the map construction a lot faster.
    """

    complete_url = url + "/construct"

    data = {
        "token": token,
        "name": mapName,
        "preservePoses": True,
        # "featureCount": 1024, #default: 1024
        # "featureCountMax": 8192, #default: 8192,
        # "featureFilter": 0, # default: 0
        # "trackLengthMin": 2, #default: 2
        # "triangulationDistanceMax": 512, # default: 512
        "dense": 0, # default: 1
    }


    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    print(r.text)
    return r.text

def main(url: str, token: str, b2gFilepath: str, mapName: str, chunk_size: int) -> None:
    """Clears the user's workspace, submits images with multithreading, and submits the map construction job to the Cloud Service"""

    ClearWorkspace(url, token)
    status_code = SubmitB2G(b2gFilepath, token, chunk_size)

    if status_code != 'error':
        StartMapConstruction(url, token, mapName)

if __name__ == '__main__':

    # Immersal international server
    url = 'https://api.immersal.com'
    token = "your_token"

    # Immersal China server
    # url = 'https://immersal.hexagon.com.cn'
    # token = "your_token"

    # Path of the '.b2g' file
    b2gFilepath = 'path/to/my/b2gFile'

    # Your map name. Please note that the map name must consist of letters or numbers (A-Z/a-z/0-9), 
    # and must not contain spaces or special characters (such as -, _, /, etc.).
    map_name = "your_map_name"

    # This is 1MB. Adjust the size as needed.
    chunk_size = 1024 * 1024  

    main(url, token, b2gFilepath, map_name, chunk_size)