# actually same as submit_navvis_json.py

from typing import List
import os
import json
import concurrent.futures
import requests                 # pip install requests
from natsort import natsorted   # pip install natsort
import numpy as np              # pip install numpy
import open3d as o3d            # pip install open3d
import cv2                      # pip install opencv-python
import math
import re

def API_clear_workspace(url: str, token: str, deleteAnchor: bool=True) -> str:
    complete_url = url + "/clear"

    data = {
        "token": token,
        "anchor": deleteAnchor,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def API_submit_image(data_list: list[dict], list_index: int, url: str, token: str) -> None:
    complete_url = url + "/capture"
    
    data = data_list[list_index]
    image_path = data["image_path"]

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = resize_image(img, longest_side=4096)

    img_bytes = cv2.imencode('.png', img)[1].tobytes()

    submit_data = {
        "token": token,
        "run": 0,
        "index": data["image_id"],
        "anchor": False,
        "px" : data["px"],
        "py" : data["py"],
        "pz" : data["pz"],
        "r00" : data["r00"],
        "r01" : data["r01"],
        "r02" : data["r02"],
        "r10" : data["r10"],
        "r11" : data["r11"],
        "r12" : data["r12"],
        "r20" : data["r20"],
        "r21" : data["r21"],
        "r22" : data["r22"],
        "fx": 0.0,
        "fy": 0.0,
        "ox": 0.0,
        "oy": 0.0
    }

    json_data = json.dumps(submit_data)
    json_bytes = json_data.encode()
    body = json_bytes + b"\0" + img_bytes
    r = requests.post(complete_url, data=body)
    return r.text


def API_start_map_construction(url: str, token: str, map_name: str) -> str:
    
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
        "name": map_name,
        "preservePoses": True,
        "featureCount": 1024, #default: 1024
        "featureCountMax": 8196, #default: 8192,
        "featureFilter": 1, # default: 0
        # "trackLengthMin": 3, #default: 2
        # "triangulationDistanceMax": 64, # default: 512
        "dense": 0, # default: 1
    }

    json_data = json.dumps(data)
    r = requests.post(complete_url, data=json_data)
    return r.text


def resize_image(img: np.ndarray, longest_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    longest_side = max(640, longest_side)

    max_side = max(h, w)
    if(longest_side <= max_side):
        resize_ratio = float(longest_side) / float(max_side)

        height, width = tuple([math.floor(x * resize_ratio) for x in img.shape[:2]])
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    return img


def setup_visualizer() -> o3d.visualization.Visualizer:

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # hex 171A1F - Immersal dark blue
    background_color = [23, 26, 31]
    normalized_background_color = [c / 255.0 for c in background_color]

    # hex DDFF19 - Immersal yellow
    point_color = [221, 255, 25]
    normalized_point_color = [c / 255.0 for c in point_color]


    opt = vis.get_render_option()
    opt.background_color = normalized_background_color
    opt.point_size = 2

    return vis


def create_x_rotation_4x4_matrix(angle: float) -> np.ndarray:
    Rx = np.array( [[ 1, 0,                              0,                             0],
                    [ 0, math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
                    [ 0, math.sin(math.radians(angle)),  math.cos(math.radians(angle)), 0],
                    [ 0, 0,                              0,                             1]])
    return Rx

def create_y_rotation_4x4_matrix(angle: float) -> np.ndarray:
    Ry = np.array(  [[math.cos(math.radians(angle)),    0,  math.sin(math.radians(angle)),  0],
                     [0,                                1,  0,                              0],
                     [-math.sin(math.radians(angle)),   0,  math.cos(math.radians(angle)),  0],
                     [0,                                0,  0,                              1]])
    return Ry


def convert_coordinate_system(data_list: list[dict]) -> list[dict]:

    for data in data_list:
        px = data["px"]
        py = data["py"]
        pz = data["pz"]
        r00 = data["r00"]
        r01 = data["r01"]
        r02 = data["r02"]
        r10 = data["r10"]
        r11 = data["r11"]
        r12 = data["r12"]
        r20 = data["r20"]
        r21 = data["r21"]
        r22 = data["r22"]

        m = np.array([[r00, -r02, r01, 0],
                      [r10, -r12, r11, 0],
                      [r20, -r22, r21, 0],
                      [0,    0,   0,   1 ]])
        
        Ry = create_y_rotation_4x4_matrix(90)
        m  = np.matmul(m, Ry)

        # rotate the full pose with position and rotation -90 degrees on X-axis around the world origin to convert from Z-up to Y-up world coordinates
        p = [px, py, pz]
        m[:3, 3] = p
        Rx = create_x_rotation_4x4_matrix(-90)
        m = np.matmul(Rx, m)

        data["px"] = m[0][3]
        data["py"] = m[1][3]
        data["pz"] = m[2][3]
        data["r00"] = m[0][0]
        data["r01"] = m[0][1]
        data["r02"] = m[0][2]
        data["r10"] = m[1][0]
        data["r11"] = m[1][1]
        data["r12"] = m[1][2]
        data["r20"] = m[2][0]
        data["r21"] = m[2][1]
        data["r22"] = m[2][2]

    return data_list


def parse_trajectory_json(directory: str) -> list[dict]:
    data_list = []
    json_files = [os.path.join(directory, file) for file in natsorted(os.listdir(directory)) if file.endswith('.json')]

    for index, json_file in enumerate(json_files):
        with open(json_file, 'r') as file:
            data = json
        with open(json_file, 'r') as file:
            data = json.load(file)
            data['image_id'] = index  # 添加 image_id 键
            data_list.append(data)

    data_list = convert_coordinate_system(data_list)
    return data_list


def append_data_with_image_path(data_list: list[dict], directory: str) -> list[dict]:
    
    for data in data_list:
        image_path = os.path.join(directory, f"{data['img']}.jpg")
        
        if not os.path.exists(image_path):
            print(f"Warning: can't find image {image_path}")
            return

        data["image_path"] = image_path

    return data_list


def visualize_data(data_list: list[dict], every_nth: int=1, show_origin: bool=True) -> None:
    vis = setup_visualizer()

    every_nth = max(1, every_nth)
    for data in data_list[::every_nth]:
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        px = data["px"]
        py = data["py"]
        pz = data["pz"]

        T = np.array([[data["r00"],  data["r01"],  data["r02"],  px],
                      [data["r10"],  data["r11"],  data["r12"],  py],
                      [data["r20"],  data["r21"],  data["r22"],  pz],
                      [0,            0,            0,            1 ]])

        mesh.transform(T)
        vis.add_geometry(mesh)

    if show_origin:
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
        vis.add_geometry(origin)

    vis.run()
    vis.destroy_window()


def set_root_to_origin(data_list: list[dict]) -> list[dict]:
    first_pos = [
        data_list[0]["px"],
        data_list[0]["py"],
        data_list[0]["pz"]
    ]

    for data in data_list:
        px = data["px"] - first_pos[0]
        py = data["py"] - first_pos[1]
        pz = data["pz"] - first_pos[2]

        data["px"] = px
        data["py"] = py
        data["pz"] = pz

    return data_list


def is_valid_map_name(input_name: str) -> bool:
    match = re.compile(r'^[a-zA-Z0-9]+$')
    if len(re.findall(match, input_name)) != 1 or len(input_name) < 3:
        return False
    return True


def submit_job(data_list: list[dict], url: str, token: str, map_name: str, max_threads: int=6) -> None:
    
    r = API_clear_workspace(url, token)
    print(f"Info: cleared workspace, error: {json.loads(r)['error']}")

    nb_data = len(data_list)
    curr = 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        results = [executor.submit(API_submit_image, data_list, i, url, token) for i in range(0, nb_data)]
        for f in concurrent.futures.as_completed(results):
            try:
                r = f.result()
                print(f"Info: submitted image {curr}/{nb_data}, result: {r}")
            except Exception as e:
                print(f"Error: {e}")
            curr += 1

    r = API_start_map_construction(url, token, map_name)
    print(f"Info: started map construction, error: {json.loads(r)['error']}")


def main(directory: str, url: str, token: str, map_name: str) -> None:

    # guard clauses
    if not is_valid_map_name(map_name):
        print(f"invalid map name: {map_name}")
        return

    # prepare all data
    data_list = parse_trajectory_json(directory)
    data_list = append_data_with_image_path(data_list, directory)
    data_list = set_root_to_origin(data_list)

    # clear workspace, submit images, start map construction
    submit_job(data_list, url, token, map_name)

    # visualize poses with open3d
    visualize_data(data_list, every_nth=1, show_origin=True)


if __name__ == '__main__':
    dir_of_images_and_json = r"<path_to_the_images_and_json_files>"
    
    url = 'https://api.immersal.com/'
    token = "<your_token>"

    # url = 'https://immersal.hexagon.com.cn/'
    # token = "your_token"

    map_name = '<map_name>'

    main(dir_of_images_and_json, url, token, map_name)
