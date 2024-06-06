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
    """Returns any errors or the new map's id and size (number of images) on the Cloud Service as a JSON formatted string. Starts the map construction job for the images in the user's workspace.
    
    featureCount    Integer for the max amount of features per image. Increases the total amount of features in the map but also increases map filesize
    preservePoses   Boolean to enable constraints input images' camera pose data as constraints. Speeds up map construction if input pose data is accurate
    """

    complete_url = url + "/construct"

    data = {
        "token": token,
        "name": mapName,
        "preservePoses": True,

        # By increasing these value (usually 'featureCountMax' more feature will be extracted, but it will also generate bigger map.
        # "featureCount": 1024,
        # "featureCountMax": 8192,

        # By setting to '1' would be benefitial for high-frequency feature environemnt such as greeneries. 
        # "featureFilter": 0,

        # By increasing this number, it will leave less point (but more reliable points) in the map, so that map size can get decreased. 
        # "trackLengthMin": 2,

        # This parameter refer to the max distance of target object. You may adjust according to your environment.
        # "triangulationDistanceMax": 512

        # by setting this parameter to 0, it will skip the dense map and glb file generation, which can make the map construction a lot faster.
        "dense": 0,
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