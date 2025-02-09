from typing import List
import os
import concurrent.futures
import json
from natsort import natsorted   # pip install natsort
import requests                 # pip install requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # pip install urllib3
import signal
import sys


# Define a session with retry strategy
def requests_retry_session(
    retries=5,
    backoff_factor=0.5,
    status_forcelist=(500, 502, 504, 429),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


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

    r = requests_retry_session().post(complete_url, data=json_data, timeout=30)
    print(r.text)
    return r.text


def SubmitImage(url: str, token: str, imagesAndPoses: List[dict], index: int) -> str:
    """Returns any errors or the path of the submitted image in the Cloud Service as a JSON formatted string. Submits an image to the user's workspace. Binary version of the endpoint for faster image uploads.

    Takes in a list of dictionaries containing the .png image filepath and .json pose filepath.

    run         Integer for the tracker session. increment if tracking is lost or image is from a different session
    index       Running integer index for the images in the same run
    anchor      Boolean to flag an anchor image. Only one can exist in the map. Defines map origin and Z-axis direction
    px, py, pz  Float values for the image's camera position. Used to always set metric scale and also as constraints if preservePoses is True in a REST API construct request
    r00-r22     Float values for a 3x3 matrix of the image's camera rotation
    fx, fy      Float values for the horizontal and vertical pixel focal length for the image
    ox, oy      Float values for the principal point offset for the image
    latitude    Float value for the WGS84 latitude for the image's camera position (optional)
    longitude   Float value for the WGS84 longitude for the image's camera position (optional)
    altitude    Float value for the WGS84 altitude for the image's camera position (optional)
    img_bytes   Bytes containing the image as a .png file
    """

    complete_url = url + "/capture"

    try:
        # File existence check
        if not os.path.exists(imagesAndPoses[index]['image']):
            return {'error': f"Image file not found: {imagesAndPoses[index]['image']}", 'imageIndex': index}
        if not os.path.exists(imagesAndPoses[index]['pose']):
            return {'error': f"Pose file not found: {imagesAndPoses[index]['pose']}", 'imageIndex': index}

        with open(imagesAndPoses[index]['image'], 'rb') as imgFile:
            img_bytes = imgFile.read()

            with open(imagesAndPoses[index]['pose'], 'r') as jsonFile:
                jsonData = json.load(jsonFile)

                # JSON data integrity validation
                required_fields = ['px', 'py', 'pz', 'r00', 'r01', 'r02', 'r10', 'r11', 
                                 'r12', 'r20', 'r21', 'r22', 'fx', 'fy', 'ox', 'oy']
                missing_fields = [field for field in required_fields if field not in jsonData]
                if missing_fields:
                    return {'error': f"Missing fields in JSON: {missing_fields}", 'imageIndex': index}

                data = {
                    "token": token,
                    "run": 0,
                    "index": index,
                    "anchor": False,
                    "px": float(jsonData['px']),
                    "py": float(jsonData['py']),
                    "pz": float(jsonData['pz']),
                    "r00": float(jsonData['r00']),
                    "r01": float(jsonData['r01']),
                    "r02": float(jsonData['r02']),
                    "r10": float(jsonData['r10']),
                    "r11": float(jsonData['r11']),
                    "r12": float(jsonData['r12']),
                    "r20": float(jsonData['r20']),
                    "r21": float(jsonData['r21']),
                    "r22": float(jsonData['r22']),
                    "fx": float(jsonData['fx']),
                    "fy": float(jsonData['fy']),
                    "ox": float(jsonData['ox']),
                    "oy": float(jsonData['oy']),
                }

                json_data = json.dumps(data)
                json_bytes = json_data.encode()

                body = json_bytes + b"\0" + img_bytes

                try:
                    r = requests_retry_session().post(
                        complete_url, 
                        data=body, 
                        timeout=(30, 180),
                        headers={'Content-Type': 'application/octet-stream'}
                    )
                    r.raise_for_status()
                    j = json.loads(r.text)
                    j["imageIndex"] = index
                    return j
                except requests.exceptions.RequestException as e:
                    print(f"Error uploading image {index}: {e}")
                    if hasattr(e.response, 'text'):
                        print(f"Server response: {e.response.text}")
                    return {'error': str(e), 'imageIndex': index}

    except Exception as e:
        return {'error': f"Unexpected error: {str(e)}", 'imageIndex': index}


def StartMapConstruction(url: str, token: str, mapName: str, preservePoses: bool=False) -> str:
    """Returns any errors or the new map's id and size (number of images) on the Cloud Service as a JSON formatted string. Starts the map construction job for the images in the user's workspace.
    
    featureCount    Integer for the max amount of features per image. Increases the total amount of features in the map but also increases map filesize
    preservePoses   Boolean to enable constraints input images' camera pose data as constraints. Speeds up map construction if input pose data is accurate
    """

    complete_url = url + "/construct"

    data = {
        "token": token,
        "name": mapName,
        "preservePoses": preservePoses,

        # By increasing these value (usually 'featureCountMax' more feature will be extracted, but it will also generate bigger map.
        # "featureCount": 1024,
        # "featureCountMax": 8192,

        # By setting to '1' would be benefitial for high-frequency feature environemnt such as greeneries. 
        # "featureFilter": 1,

        # By increasing this number, it will leave less point (but more reliable points) in the map, so that map size can get decreased. 
        # "trackLengthMin": 2,

        # This parameter refer to the max distance of target object. You may adjust according to your environment.
        # "triangulationDistanceMax": 64,

        # by setting this parameter to 0, it will skip the dense map and glb file generation, which can make the map construction a lot faster.
        "dense": 0,
    }

    json_data = json.dumps(data)

    r = requests_retry_session().post(complete_url, data=json_data, timeout=30)
    print(r.text)
    return r.text


def main(url: str, token: str, inputDirectory: str, mapName: str, preservePoses: bool=False, maxThreads: int=1) -> None:
    """Clears the user's workspace, submits images with multithreading, and submits the map construction job to the Cloud Service"""

    # Add signal handler
    def signal_handler(signum, frame):
        print("\nInterrupt signal detected, force stopping...")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)

    ClearWorkspace(url, token)

    jsonFiles = [os.path.join(inputDirectory, file) for file in natsorted(os.listdir(inputDirectory)) if file.endswith('.json')]
    imagesAndPoses = []

    totalImages = len(jsonFiles)
    maxChars = len(str(totalImages))

    # Supported image extensions
    supported_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']

    for j in jsonFiles:
        with open(j, 'r') as jsonFile:
            jsonData = json.load(jsonFile)
            base_image_path = os.path.join(inputDirectory, jsonData['img'])
            
            # Try to find matching image file
            image_path = None
            # 1. First check if complete path image file exists
            if os.path.exists(base_image_path):
                image_path = base_image_path
            else:
                # 2. Try all supported extensions
                for ext in supported_extensions:
                    test_path = base_image_path + ext
                    if os.path.exists(test_path):
                        image_path = test_path
                        break
            
            if image_path is None:
                print(f"Warning: No supported format found for image file {jsonData['img']}")
                continue
                
            x = {'image': image_path, 'pose': j}
            imagesAndPoses.append(x)

    executor = None
    try:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=maxThreads)
        futures = []
        
        # Submit all tasks
        futures = [executor.submit(SubmitImage, url, token, imagesAndPoses, i) 
                  for i in range(0, len(imagesAndPoses))]
        
        # Process results
        for f in concurrent.futures.as_completed(futures):
            try:
                r = f.result(timeout=1)  # Add timeout
                curr_i = str(r["imageIndex"]+1).zfill(maxChars)
                print(f'{curr_i}/{totalImages}')
                print(r)
            except concurrent.futures.TimeoutError:
                continue
            except KeyboardInterrupt:
                raise

        # Only construct map after all uploads are successful
        StartMapConstruction(url, token, mapName, preservePoses)

    except (KeyboardInterrupt, SystemExit):
        print("\nForce stopping all tasks...")
        if executor:
            executor._threads.clear()
            concurrent.futures.thread._threads_queues.clear()
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if executor and not executor._shutdown:
            executor.shutdown(wait=False, cancel_futures=True)


if __name__ == '__main__':

    # Immersal international server
    url = 'https://api.immersal.com'
    token = "<your-token>"

    # Immersal China server
    # url = 'https://immersal.hexagon.com.cn'
    # token = "<your-token>"

    # Path of your frames, which should include both images and camera poses
    inputDirectory = r"<your-frames-directory>"

    # Your map name. Please note that the map name must consist of letters or numbers (A-Z/a-z/0-9), 
    # and must not contain spaces or special characters (such as -, _, /, etc.).
    mapName = "<your-map-name>"
    
    main(url, token, inputDirectory, mapName, preservePoses=True, maxThreads=4)
