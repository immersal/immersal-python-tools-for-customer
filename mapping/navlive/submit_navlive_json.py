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

    img_bytes = cv2.imencode('.jpg', img)[1].tobytes()

    print(
    f"Image ID: {data['image_id']} | "
    f"fx: {data['fx']}, fy: {data['fy']}, ox: {data['ox']}, oy: {data['oy']} | "
    f"Rotation:\n"
    f"  [{data['r00']:.4f}, {data['r01']:.4f}, {data['r02']:.4f}]\n"
    f"  [{data['r10']:.4f}, {data['r11']:.4f}, {data['r12']:.4f}]\n"
    f"  [{data['r20']:.4f}, {data['r21']:.4f}, {data['r22']:.4f}]\n"
    f"Translation: px: {data['px']:.4f}, py: {data['py']:.4f}, pz: {data['pz']:.4f}"
)


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
    complete_url = url + "/construct"

    data = {
        "token": token,
        "name": map_name,
        "preservePoses": True,

        # By increasing these value (usually 'featureCountMax' more feature will be extracted, but it will also generate bigger map.
        # "featureCount": 1024,
        # "featureCountMax": 8192,

        # By setting to '1' would be benefitial for high-frequency feature environemnt such as greeneries. 
        "featureFilter": 1,

        # By increasing this number, it will leave less point (but more reliable points) in the map, so that map size can get decreased. 
        # "trackLengthMin": 2,

        # This parameter refer to the max distance of target object. You may adjust according to your environment.
       # "triangulationDistanceMax": 1024,

        # by setting this parameter to 0, it will skip the dense map and glb file generation, which can make the map construction a lot faster.
        "dense": 0,
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

        m = np.eye(4)
        m[:3, :3] = np.array([
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22]
        ])
        m[:3, 3] = [px, py, pz]


        Rx = create_x_rotation_4x4_matrix(-90)
        m = Rx @ m

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
        img_name = data['img']
        if not img_name.lower().endswith(('.jpg', '.jpeg')):
            img_name += '.jpg'  

        image_path = os.path.join(directory, img_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: can't find image {image_path}")
            continue

        data["image_path"] = image_path

    return data_list

def append_data_with_image_path_filtered(data_list: list[dict], directory: str, skip_keyword: str = "") -> list[dict]:
    filtered_list = []

    for data in data_list:
        img_name = data['img']
        
        # Skip if the image name contains the skip keyword
        if skip_keyword and skip_keyword.lower() in img_name.lower():
            print(f"Skipping image due to keyword match: {img_name}")
            continue

        # Ensure correct file extension
        if not img_name.lower().endswith(('.jpg', '.jpeg')):
            img_name += '.jpg'  # or '.jpg' depending on your convention

        image_path = os.path.join(directory, img_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: can't find image {image_path}")
            continue

        data["image_path"] = image_path
        filtered_list.append(data)

    return filtered_list



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
    if not is_valid_map_name(map_name):
        print(f"Invalid map name: {map_name}")
        return

    data_list = []

    for subfolder in sorted(os.listdir(directory)):
        subpath = os.path.join(directory, subfolder)
        if not os.path.isdir(subpath):
            continue

        print(f"Processing folder: {subpath}")
        sub_data = parse_trajectory_json(subpath)
        if not sub_data:
            print(f"Warning: no JSON data found in {subpath}")
            continue

       # sub_data = append_data_with_image_path(sub_data, subpath)
        sub_data = append_data_with_image_path_filtered(sub_data, subpath, "combined")
        if not sub_data:
            print(f"Warning: image paths could not be resolved in {subpath}")
            continue

        data_list.extend(sub_data)

    if not data_list:
        print("Error: No data found in any subfolder.")
        return

    # data_list = set_root_to_origin(data_list)

    submit_job(data_list, url, token, map_name)

    visualize_data(data_list, every_nth=1, show_origin=True)


if __name__ == '__main__':
    url = 'https://api.immersal.com/'
    directory = r"your_image_directory"

    token = "your_token"
    
    map_name = 'your_map_name'

    main(directory, url, token, map_name)
