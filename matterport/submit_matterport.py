from typing import List
import os
import math
import json
from dataclasses import dataclass
from typing import List, Optional
from natsort import natsorted   # pip install natsort
import numpy as np              # pip install numpy
import open3d as o3d            # pip install open3d
import requests                 # pip install requests
import cv2                      # pip install opencv-python

def clear_workspace(delete_anchor: bool=True) -> str:
    complete_url = url + "/clear"

    data = {
        "token": token,
        "anchor": delete_anchor,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    print(r.text)
    return r.text

def submit_image(index: int, image_path: str, image_pose: np.ndarray, intrinsics: np.ndarray, resize_factor: float=1.0) -> str:
    complete_url = url + "/capture"

    with open(image_path, 'rb') as image_file:
        img_bytes = image_file.read()

        if(resize_factor != 1.0):
            nparray = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)

            height, width, channel = img.shape

            new_height = math.floor(height * resize_factor)
            new_width = math.floor(width * resize_factor)

            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            img_bytes = cv2.imencode('.png', resized_img)[1].tobytes()

        data = {
            "token": token,
            "run": 0,
            "index": index,
            "anchor": False,
            "px": image_pose[0][3],
            "py": image_pose[1][3],
            "pz": image_pose[2][3],
            "r00": image_pose[0][0],
            "r01": image_pose[0][1],
            "r02": image_pose[0][2],
            "r10": image_pose[1][0],
            "r11": image_pose[1][1],
            "r12": image_pose[1][2],
            "r20": image_pose[2][0],
            "r21": image_pose[2][1],
            "r22": image_pose[2][2],
            "fx": intrinsics[0] * resize_factor,
            "fy": intrinsics[1] * resize_factor,
            "ox": intrinsics[2] * resize_factor,
            "oy": intrinsics[3] * resize_factor,
        }

        json_data = json.dumps(data)
        json_bytes = json_data.encode()

        body = json_bytes + b"\0" + img_bytes

        r = requests.post(complete_url, data=body)
        print(r.text)
        return r.text

@dataclass
class MapConstructionParams:
    featureCount: Optional[int] = 1024
    preservePoses: Optional[bool] = True

def construct_map(params: MapConstructionParams) -> str:
    complete_url = url + "/construct"

    data = {
        "token": token,
        "name": map_name,
        "featureCount": params.featureCount,
        "preservePoses": params.preservePoses,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    print(r.text)
    return r.text

def setup_visualizer() -> o3d.visualization.Visualizer:

    # hex 171A1F - Immersal dark blue
    background_color = [23, 26, 31]
    normalized_background_color = [c / 255.0 for c in background_color]

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = normalized_background_color
    opt.point_size = 1

    return vis

# returns true if any point is within distance_treshold
# ignores cases where distance <= min_distance (to ignore overlapping poses)
def has_point_within(point: np.ndarray, points: List[np.ndarray], distance_threshold: float, min_distance: float = 0.01):
    for p in points:
        distance = np.sqrt(np.sum((p - point)**2))
        if distance <= min_distance:
            continue
        if distance <= distance_threshold:
            print(f"Found another pose at distance = {distance}")
            return True
    return False

@dataclass
class ProcessParams:
    add_origin: Optional[bool] = False,
    coordinate_frame_size: Optional[float] = 0.3
    ply_path: Optional[str] = ''
    submit: Optional[bool] = False
    img_resize_factor: Optional[float] = 0.5
    pose_distance_threshold: Optional[float] = -1

def process_poses(images_and_poses: List[dict], params: ProcessParams, map_params: MapConstructionParams) -> None:

    # bounding box for debugging
    bb_min = [math.inf, math.inf, math.inf]
    bb_max = [-math.inf, -math.inf, -math.inf]

    vis = setup_visualizer()

    if(params.submit):
        clear_workspace()

    points = []

    for i, x in enumerate(images_and_poses):
        print(f"\nReading: {(x['pose'])}")
        with open(x['pose'], 'r') as json_file:
            data = json.load(json_file)

            px = data['px']
            py = data['py']
            pz = data['pz']

            if (params.pose_distance_threshold != -1):
                # skip poses that are too close to existing ones
                # (ignores poses within min_distance=0.01 to not skip overlapping)
                if has_point_within(np.array([px,py,pz]), points, distance_threshold=params.pose_distance_threshold):
                    print("Pose too close to other poses, skipping.\n")
                    continue

            points.append(np.array([px,py,pz]))

            r00 = data['r00']
            r01 = data['r01']
            r02 = data['r02']
            r10 = data['r10']
            r11 = data['r11']
            r12 = data['r12']
            r20 = data['r20']
            r21 = data['r21']
            r22 = data['r22']

            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=params.coordinate_frame_size, origin=[0, 0, 0])

            # invert Y & Z
            T = np.array([  [r00,  -r01,  -r02,  px],
                            [r10,  -r11,  -r12,  py],
                            [r20,  -r21,  -r22,  pz],
                            [  0,    0,     0,   1]])

            # -90 deg around global X
            TD = np.array([ [1,  0,  0,  0],
                            [0,  0,  1,  0],
                            [0, -1,  0,  0],
                            [0,  0,  0,  1]])
            
            T = np.dot(TD, T)

            mesh.transform(T)
            vis.add_geometry(mesh)

            if(params.submit):
                intrinsics = np.array([data['fx'], data['fy'], data['ox'], data['oy']])
                submit_image(i, x['image'], T, intrinsics, resize_factor=params.img_resize_factor)

            with np.printoptions(precision=3, suppress=True):
                print(T)

            bb_min = [min(x, bb_min[i]) for i, x in enumerate(T[0:3, 3])]
            bb_max = [max(x, bb_max[i]) for i, x in enumerate(T[0:3, 3])]

    print(f"Processed {len(points)} poses.\n")

    if(params.add_origin):
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
        vis.add_geometry(origin)

    if os.path.exists(params.ply_path):
        pcd = o3d.io.read_point_cloud(params.ply_path, format='ply')
        # pcd, ind = pcd.remove_radius_outlier(nb_points=16, radius=2.0)
        vis.add_geometry(pcd)

    # bounding box for debugging 
    with np.printoptions(precision=2, suppress=True):
        print(f'bb_min:\t{np.array(bb_min)}')
        print(f'bb_max:\t{np.array(bb_max)}')

    if(params.submit):
        construct_map(map_params)

    vis.run()
    vis.destroy_window()

def main(input_directory: str, process_params: ProcessParams, map_params: MapConstructionParams) -> None:
    json_files = []
    dirs = natsorted(os.listdir(input_directory))
    for i, dir in enumerate(dirs):
        dir_path = os.path.join(input_directory, dir)
        if os.path.isdir(dir_path):
            for file in natsorted(os.listdir(dir_path)):
                if file.endswith('.json'):
                    json_files.append(os.path.join(dir_path, file))

    images_and_poses = []

    for j in json_files:
        with open(j, 'r') as json_file:
            json_data = json.load(json_file)
            image_path = os.path.join(os.path.dirname(j), f"{json_data['img']}.jpg")
            x = {'image': image_path, 'pose': j}
            images_and_poses.append(x)

    process_poses(images_and_poses, process_params, map_params)

if __name__ == '__main__':

    # 1. Install all dependencies mentioned at the top of the file with pip
    # 2. Acquire an e57 file of the desired Matterport scan
    # 3. Use the unpack_matterport_e57.py script to unpack the data
    # 4. Set input_directory to point to the directory where the data was unpacked
    # 5. Give a name for your map in map_name and input your Immersal Developer Token in token
    # 6. Run the script and go check Immersal Develop Portal if no errors were presented

    global url
    global token
    global map_name

    # Immersal international server
    url = 'https://api.immersal.com'
    token = "your_token"

    # Immersal China server
    # url = 'https://immersal.hexagon.com.cn'
    # token = "your_token"

    map_name = "matterport_map"

    # Path of your Matterport scan output
    input_directory = r"/data/matterport/my_matterport_scan-out/"

    # set submit to False to only visualize poses
    process_params = ProcessParams(submit=True)
    map_params = MapConstructionParams()

    main(input_directory, process_params, map_params)