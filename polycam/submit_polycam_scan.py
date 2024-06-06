from typing import List, Union
import os
import concurrent.futures
import json
import re
import numpy as np              # pip install numpy
import cv2                      # pip install opencv-python
import requests                 # pip install requests


class PolycamScanArchive():

    def is_optimized(self) -> bool:
        if not os.path.exists(os.path.join(self.archive_path, "keyframes/corrected_cameras")):
            return False
        if not os.path.exists(os.path.join(self.archive_path, "keyframes/corrected_images")):
            return False
        return True

    def contains_data(self) -> bool:
        if not os.path.isfile(self.mesh_info_path):
            return False
        if len(self.cameras) < 1:
            return False
        if len(self.images) < 1:
            return False
        return True

    def __init__(self, archive_path: str):
        self.archive_path = archive_path

        if self.is_optimized():
            self.cameras_path = os.path.join(archive_path, "keyframes/corrected_cameras")
            self.images_path = os.path.join(archive_path, "keyframes/corrected_images")
        else:
            self.cameras_path = os.path.join(archive_path, "keyframes/cameras")
            self.images_path = os.path.join(archive_path, "keyframes/images")

        self.mesh_info_path = os.path.join(self.archive_path, "mesh_info.json")

        self.cameras = [os.path.join(self.cameras_path, file) for file in os.listdir(self.cameras_path) if file.endswith(".json")]
        self.images = [os.path.join(self.cameras_path, file) for file in os.listdir(self.images_path) if file.endswith(".jpg") or file.endswith(".png")]
        self.contains_data = self.contains_data()

        if self.contains_data:
            with open(self.mesh_info_path, 'r') as mesh_info_json:
                mesh_info_data = json.load(mesh_info_json)

                t_00 = mesh_info_data['alignmentTransform'][0]
                t_01 = mesh_info_data['alignmentTransform'][4]
                t_02 = mesh_info_data['alignmentTransform'][8]
                t_03 = mesh_info_data['alignmentTransform'][12]
                t_10 = mesh_info_data['alignmentTransform'][1]
                t_11 = mesh_info_data['alignmentTransform'][5]
                t_12 = mesh_info_data['alignmentTransform'][9]
                t_13 = mesh_info_data['alignmentTransform'][13]
                t_20 = mesh_info_data['alignmentTransform'][2]
                t_21 = mesh_info_data['alignmentTransform'][6]
                t_22 = mesh_info_data['alignmentTransform'][10]
                t_23 = mesh_info_data['alignmentTransform'][14]
                t_30 = mesh_info_data['alignmentTransform'][3]
                t_31 = mesh_info_data['alignmentTransform'][7]
                t_32 = mesh_info_data['alignmentTransform'][11]
                t_33 = mesh_info_data['alignmentTransform'][15]

                # 4x4 transform matrix for Polycam mesh Alignment
                T = np.array([ [t_00,  t_01,  t_02, t_03],
                                [t_10,  t_11,  t_12, t_13],
                                [t_20,  t_21,  t_22, t_23],
                                [t_30,  t_31,  t_32, t_33]])

                self.mesh_transform = T


class Keyframe():
    def __init__(self, json_path: str, scan_data: PolycamScanArchive):

        self.is_valid = True

        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)

        required_keys = [   "cx",
                            "cy",
                            "fx",
                            "fy",
                            "timestamp",
                            "t_00",
                            "t_01",
                            "t_02",
                            "t_03",
                            "t_10",
                            "t_11",
                            "t_12",
                            "t_13",
                            "t_20",
                            "t_21",
                            "t_22",
                            "t_23",
            ]
            
        for k in required_keys:
            if type(json_data.get(k)) not in [float, int]:
                self.is_valid = False
                print(f"WARNING: {json_path} does not contain valid required camera metadata values")

        if self.is_valid:       
            self.fx = json_data.get("fx")
            self.fy = json_data.get("fy")
            self.ox = json_data.get("cx")
            self.oy = json_data.get("cy")

            self.timestamp = json_data.get("timestamp")
            self.image_path = os.path.join(scan_data.images_path, f"{str(self.timestamp)}.jpg")

            t_00 = json_data.get("t_00")
            t_01 = json_data.get("t_01")
            t_02 = json_data.get("t_02")
            t_03 = json_data.get("t_03")
            t_10 = json_data.get("t_10")
            t_11 = json_data.get("t_11")
            t_12 = json_data.get("t_12")
            t_13 = json_data.get("t_13")
            t_20 = json_data.get("t_20")
            t_21 = json_data.get("t_21")
            t_22 = json_data.get("t_22")
            t_23 = json_data.get("t_23")

            # Negating rotation Y and Z axes to match Immersal coordinate system
            self.local_position = np.array([t_03, t_13, t_23])
            self.local_rotation = np.array([t_00, -t_01, -t_02, t_10, -t_11, -t_12, t_20, -t_21, -t_22])

            # 4x4 transform matrix of the above
            T_local = np.array([[t_00,   -t_01,   -t_02,    t_03],
                                [t_10,   -t_11,   -t_12,    t_13],
                                [t_20,   -t_21,   -t_22,    t_23],
                                [0,       0,       0,       1   ]])

            # final 4x4 transform matrix: camera poses multiplied by the Polycam mesh alignment
            T = np.matmul(scan_data.mesh_transform, T_local)

            # extract rotation and position components from the final 4x4 transform matrix and flatten to lists
            self.position = T[:3,3:].flatten()
            self.rotation = T[:3,:3].flatten()


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


def SubmitImage(url: str, token: str, keyframe_list: List[dict], index: int) -> str:
    """Returns any errors or the path of the submitted image in the Cloud Service as a JSON formatted string. Submits an image to the user's workspace. Binary version of the endpoint for faster image uploads.

    Takes in a list of dictionaries containing all image data needed to submit them.

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

    keyframe = keyframe_list[index]

    img = cv2.imread(keyframe.image_path, cv2.IMREAD_GRAYSCALE)
    img_bytes = cv2.imencode('.png', img)[1].tobytes()

    data = {
        "token": token,
        "run": 0,
        "index": index,
        "anchor": False,
        "px": keyframe.position[0],
        "py": keyframe.position[1],
        "pz": keyframe.position[2],
        "r00": keyframe.rotation[0],
        "r01": keyframe.rotation[1],
        "r02": keyframe.rotation[2],
        "r10": keyframe.rotation[3],
        "r11": keyframe.rotation[4],
        "r12": keyframe.rotation[5],
        "r20": keyframe.rotation[6],
        "r21": keyframe.rotation[7],
        "r22": keyframe.rotation[8],
        "fx": keyframe.fx,
        "fy": keyframe.fy,
        "ox": keyframe.ox,
        "oy": keyframe.oy,
        "latitude" : "",
        "longitude" : "",
        "altitude" : "",
    }

    json_data = json.dumps(data)
    json_bytes = json_data.encode()

    body = json_bytes + b"\0" + img_bytes

    r = requests.post(complete_url, data=body)
    return r.text


def ValidateMapName(input_name:str) -> bool:
    match = re.compile(r'^[a-zA-Z0-9]+$')
    if len(re.findall(match, input_name)) != 1 or len(input_name) < 3:
        return False
    return True



def main(url: str, token: str, map_name: str, input_directory: str, max_threads: int=4):

    if not ValidateMapName(map_name):
        print(f"invalid map name: {map_name}, must be 3 or more [a-zA-A0-9] characters")
        return

    polycam_scan = PolycamScanArchive(input_directory)

    if not polycam_scan.contains_data:
        print(f"ERROR: scan in {input_directory} does not contain necessary data")
        return

    keyframes = []
    
    for c in polycam_scan.cameras:
        kf = Keyframe(c, polycam_scan)
        keyframes.append(kf)

    # clear the user's workspace from all existing images
    ClearWorkspace(url, token)


    # submit all images from Polycam scan
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        results = [executor.submit(SubmitImage, url, token, keyframes, i) for i in range(0, len(keyframes))]

        for f in concurrent.futures.as_completed(results):
            print(f.result())

    # start map construction
    StartMapConstruction(url, token, map_name)
    print(f"Map {map_name} submitted to Immersal Cloud Service")


if __name__ == '__main__':

    # https://github.com/PolyCam/polyform

    # 0. Install all dependancies mentioned at the top of the file with pip
    # 1. Enable Developer mode in Polycam Settings
    # 2. Scan something and download .zip of the scan from export/other/raw data (images, poses, depth maps)
    # 3. Unzip somewhere on your computer and input the path to the directory to input_directory
    # 4. Give a name for your map in map_name and input your Immersal Developer Token in token
    # 5. Run the script and go check Immersal Develop Portal if no errors were presented

    # Immersal international server
    url = 'https://api.immersal.com'
    token = "your_token"

    # Immersal China server
    # url = 'https://immersal.hexagon.com.cn'
    # token = "your_token"

    # Path of the input directory, which is the root directory of Polycam export, which is the parant 
    # directory of folder 'keyframes'
    # e.g. /home/maolin/workspace/mapping-data-polycam/garden/Sep1at10-00AM
    input_directory = "path_of_polycam_export"

    # Your map name. Please note that the map name must consist of letters or numbers (A-Z/a-z/0-9), 
    # and must not contain spaces or special characters (such as -, _, /, etc.).
    map_name = "your_map_name"

    main(url, token, map_name, input_directory)

