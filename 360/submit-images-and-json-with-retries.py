from typing import List
import os
import concurrent.futures
import json
from natsort import natsorted   # pip install natsort
import requests                 # pip install requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # pip install urllib3


# Define a session with retry strategy
def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
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

    with open(imagesAndPoses[index]['image'], 'rb') as imgFile:
        img_bytes = imgFile.read()

        with open(imagesAndPoses[index]['pose'], 'r') as jsonFile:
            jsonData = json.load(jsonFile)

            data = {
                "token": token,
                "run": 0,
                "index": index,
                "anchor": False,
                "px": jsonData['px'],
                "py": jsonData['py'],
                "pz": jsonData['pz'],
                "r00": jsonData['r00'],
                "r01": jsonData['r01'],
                "r02": jsonData['r02'],
                "r10": jsonData['r10'],
                "r11": jsonData['r11'],
                "r12": jsonData['r12'],
                "r20": jsonData['r20'],
                "r21": jsonData['r21'],
                "r22": jsonData['r22'],
                "fx": jsonData['fx'],
                "fy": jsonData['fy'],
                "ox": jsonData['ox'],
                "oy": jsonData['oy'],
            }

            json_data = json.dumps(data)
            json_bytes = json_data.encode()

            body = json_bytes + b"\0" + img_bytes

            try:
                r = requests_retry_session().post(complete_url, data=body, timeout=60)
                r.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Error uploading image {index}: {e}")
                return {'error': str(e), 'imageIndex': index}
            
            j = json.loads(r.text)
            j["imageIndex"] = index

            return j


def StartMapConstruction(url: str, token: str, mapName: str, preservePoses: bool=False) -> str:

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
        "preservePoses": preservePoses,
        # "featureCount": 1024, #default: 1024
        # "featureCountMax": 8192, #default: 8192,
        # "featureFilter": 0, # default: 0
        # "trackLengthMin": 2, #default: 2
        # "triangulationDistanceMax": 512, # default: 512
        "dense": 0, # default: 1
    }

    json_data = json.dumps(data)

    r = requests_retry_session().post(complete_url, data=json_data, timeout=30)
    print(r.text)
    return r.text


def main(url: str, token: str, inputDirectory: str, mapName: str, preservePoses: bool=False, maxThreads: int=1) -> None:
    """Clears the user's workspace, submits images with multithreading, and submits the map construction job to the Cloud Service"""

    ClearWorkspace(url, token)

    jsonFiles = [os.path.join(inputDirectory, file) for file in natsorted(os.listdir(inputDirectory)) if file.endswith('.json')]
    imagesAndPoses = []

    totalImages = len(jsonFiles)
    maxChars = len(str(totalImages))

    for j in jsonFiles:
        with open(j, 'r') as jsonFile:
            jsonData = json.load(jsonFile)
            imagePath = os.path.join(inputDirectory, f"{jsonData['img']}.png")
            # imagePath = os.path.join(inputDirectory, f"{jsonData['img']}.jpg")
            # imagePath = os.path.join(inputDirectory, f"{jsonData['img']}.jpeg")
            # imagePath = os.path.join(inputDirectory, f"{jsonData['img']}") #no extention name
            x = {'image': imagePath, 'pose': j}
            imagesAndPoses.append(x)


    with concurrent.futures.ThreadPoolExecutor(max_workers=maxThreads) as executor:
        results = [executor.submit(SubmitImage, url, token, imagesAndPoses, i) for i in range(0, len(imagesAndPoses))]

        for f in concurrent.futures.as_completed(results):
            r = f.result()

            curr_i = str(r["imageIndex"]+1).zfill(maxChars)
            print(f'{curr_i}/{totalImages}')
            print(r)


    # preservePoses turned on to speed up map construction and maintain the coordinates system from the input images
    StartMapConstruction(url, token, mapName, preservePoses)


if __name__ == '__main__':

    # Immersal international server
    url = 'https://api.immersal.com'
    token = "your_token"

    # Immersal China server
    # url = 'https://immersal.hexagon.com.cn'
    # token = "your_token"

    # Path of your frames, which should include both images and camera poses
    # e.g. inputDirectory = "/Users/maolin/workspaces/mapping-test-202402/tripla-rect/frames"
    inputDirectory = "path_of_frames"

    # Your map name. Please note that the map name must consist of letters or numbers (A-Z/a-z/0-9), 
    # and must not contain spaces or special characters (such as -, _, /, etc.).
    mapName = "your_map_name"

    main(url, token, inputDirectory, mapName, preservePoses=True, maxThreads=4)
