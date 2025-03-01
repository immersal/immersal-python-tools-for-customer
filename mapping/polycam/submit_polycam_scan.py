from typing import List, Union
import os
import concurrent.futures
import json
import math
import random
import copy
import colorsys
import numpy as np              # pip install numpy
import cv2                      # pip install opencv-python
import open3d as o3d            # pip install open3d
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

        # Sort the files after listing them to ensure order
        self.cameras = sorted([os.path.join(self.cameras_path, file) for file in os.listdir(self.cameras_path) if file.endswith(".json")])
        self.images = sorted([os.path.join(self.images_path, file) for file in os.listdir(self.images_path) if file.endswith(".jpg") or file.endswith(".png")])
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


def SetupVisualizer() -> o3d.visualization.Visualizer:

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # hex 171A1F - Immersal dark blue
    background_color = [23, 26, 31]
    normalized_background_color = [c / 255.0 for c in background_color]

    opt = vis.get_render_option()
    opt.background_color = normalized_background_color
    opt.point_size = 5

    return vis


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
    featureFilter               Possible values 0 or 1, in scenes where there's lots of unique details like grass, bushes, leaves, gravel etc we'd pick a lot of noisy details (high frequency features) as our top picks. These features were very hard to localize against. By using this parameter in map construction, it sorts the detected features based on size favoring large features (low frequency features). This seems to improve map construction and localization rate in high frequency environments.
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


def SubmitImage(url: str, token: str, keyframe_list: List[dict], index: int, use_first_image_as_anchor: bool=True) -> str:
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
        "anchor": (index == 0) if use_first_image_as_anchor else False,
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


def distanceBetween(a: List[float], b: List[float]) -> float:
    return math.sqrt((a[0]-b[0]) ** 2 + (a[1]-b[1]) ** 2 + (a[2]-b[2]) ** 2)


def meanPosition(input_positions: List[float]) -> List[float]:
    mean_x = sum([p[0] for p in input_positions]) / float(len(input_positions))
    mean_y = sum([p[1] for p in input_positions]) / float(len(input_positions))
    mean_z = sum([p[2] for p in input_positions]) / float(len(input_positions))

    return [mean_x, mean_y, mean_z]


def split_to_groups(input_keyframes: list[Keyframe], k: int=1):
    # https://en.wikipedia.org/wiki/K-means_clustering

    centroids = {}

    # initialize k random centroids
    random_indices = random.sample(range(0, len(input_keyframes)), k)
    for i, random_index in enumerate(random_indices):
        centroids[i] = input_keyframes[random_index].position

    groups = {}

    # iterations should not be hardcoded
    for i in range(0, 50):

        g = {}
        
        for kf in input_keyframes:
            p = kf.position
            closest_centroid = 0
            closest_distance = math.inf

            for c in centroids:

                dist = distanceBetween(p, centroids[c])

                if dist < closest_distance:
                    closest_distance = dist
                    closest_centroid = c

            if g.get(closest_centroid) == None:
                g[closest_centroid] = [kf]
            else:
                g[closest_centroid].append(kf)


        for c in centroids:
            centroids[c] = meanPosition([kf.position for kf in g.get(c)])

        groups = g

    return groups, centroids


def main(url: str, token: str, map_name: str, input_directory: str, skip_submission: bool=False, visualize_poses: bool=True, coordinate_frame_size: float=0.25, centroid_frame_size: float=1.0, split_groups: int=1, max_threads: int=4, use_first_image_pose_as_origo: bool=False):

    polycam_scan = PolycamScanArchive(input_directory)

    if not polycam_scan.contains_data:
        print(f"ERROR: scan in {input_directory} does not contain necessary data")
        return

    keyframes = []
    
    for c in polycam_scan.cameras:
        kf = Keyframe(c, polycam_scan)
        keyframes.append(kf)

    keyframe_groups, centroids = split_to_groups(keyframes, k=split_groups)

    # you can skip the actual submission if you want to just visualize the splitting
    if not skip_submission:
        for g in keyframe_groups:
            split_map_name = f"{map_name}_{str(g).zfill(3)}"
            print(split_map_name)

            # clear the workspace, submit all images from Polycam scan, start map construction
            ClearWorkspace(url, token)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                results = [executor.submit(SubmitImage, url, token, keyframe_groups[g], i, use_first_image_pose_as_origo) for i in range(0, len(keyframe_groups[g]))]

                for f in concurrent.futures.as_completed(results):
                    print(f.result())

            StartMapConstruction(url, token, split_map_name)
            print(f"Map {split_map_name} submitted to Immersal Cloud Service")


    # additional image pose visualization for debugging
    if visualize_poses:
        vis = SetupVisualizer()

        # Determine what to display as the origin reference based on use_first_image_pose_as_origo parameter
        if use_first_image_pose_as_origo:
            # Display the first camera position as the origin reference
            for g in keyframe_groups:
                first_camera = keyframe_groups[g][0]
                first_position = first_camera.position
                mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=centroid_frame_size, origin=[0, 0, 0])
                mesh.translate(first_position)
                vis.add_geometry(mesh)
                print(f"Using first camera position {first_position} as origin")
        else:
            # Display the group center points as reference
            for c in centroids.values():
                mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=centroid_frame_size, origin=[0, 0, 0])
                mesh.translate(c)
                vis.add_geometry(mesh)
                print(f"Using center point {c} as origin")

        print(f"total groups: {len(keyframe_groups)}")
        for g in keyframe_groups:
            # Assign different colors to coordinate systems for each group
            group_color = colorsys.hsv_to_rgb(random.uniform(0, 1), random.uniform(0.2, 0.8), 0.7)
            
            # Get all camera positions in this group for creating paths later
            camera_positions = []
            
            # Display complete pose (position and orientation) for each camera
            camera_count = len(keyframe_groups[g])
            for i, kf in enumerate(keyframe_groups[g]):
                # Create coordinate system
                camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coordinate_frame_size, origin=[0, 0, 0])
                
                # Build camera transformation matrix (4x4)
                T = np.array([
                    [kf.rotation[0], kf.rotation[1], kf.rotation[2], kf.position[0]],
                    [kf.rotation[3], kf.rotation[4], kf.rotation[5], kf.position[1]],
                    [kf.rotation[6], kf.rotation[7], kf.rotation[8], kf.position[2]],
                    [0,              0,              0,              1              ]
                ])
                
                # Apply transformation
                camera_frame.transform(T)
                
                # Add to visualizer
                vis.add_geometry(camera_frame)
                
                # Save camera position for path creation
                camera_positions.append(kf.position)
                
                # Add sequence number labels
                # Note: Open3D doesn't directly support adding text labels, so we create a small sphere to represent the number
                # Use different colors for start and end points
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=coordinate_frame_size * 0.5)
                sphere.translate(kf.position)
                
                # Start point: green, end point: red, others: white
                if i == 0:  # start point
                    sphere.paint_uniform_color([0, 1, 0])  # green
                elif i == camera_count - 1:  # end point
                    sphere.paint_uniform_color([1, 0, 0])  # red
                else:
                    sphere.paint_uniform_color([1, 1, 1])  # white
                
                vis.add_geometry(sphere)
                
                # Display sequence information (camera index) using Open3D's simple rendering functionality
                # Note: For actual text labels, you might need other libraries like Matplotlib
                print(f"Camera {i} position: {kf.position}")
            
            # Create lines connecting camera positions to show movement path
            if len(camera_positions) > 1:
                line_points = []
                line_indices = []
                
                # Create line segments connecting points
                for i in range(len(camera_positions)-1):
                    line_points.append(camera_positions[i])
                    line_points.append(camera_positions[i+1])
                    line_indices.append([i*2, i*2+1])
                
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(line_points)
                line_set.lines = o3d.utility.Vector2iVector(line_indices)
                
                # Set line color to yellow for visibility
                colors = [[1, 1, 0] for _ in range(len(line_indices))]  # yellow
                line_set.colors = o3d.utility.Vector3dVector(colors)
                
                vis.add_geometry(line_set)
            
            # Also display point cloud for comparison
            pcd = o3d.geometry.PointCloud()
            xyz = []
            for kf in keyframe_groups[g]:
                xyz.append(kf.position)
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.paint_uniform_color(group_color)
            vis.add_geometry(pcd)

        # Add legend explanation (console output)
        print("\nCamera path visualization explanation:")
        print("- Green sphere: Starting point (Camera 1)")
        print("- Red sphere: End point (Last camera)")
        print("- White spheres: Intermediate camera positions")
        print("- Yellow lines: Camera movement path")
        print("- Colored point cloud: Camera position overview")
        print("- Small coordinate systems: Position and orientation of each camera")
        print("- Large coordinate system: Group center point")

        vis.run()
        vis.destroy_window()


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
    token = "<your_token>"

    # Immersal China server
    # url = 'https://immersal.hexagon.com.cn'
    # token = "your_token"

    # Path of the input directory, which is the root directory of Polycam export, which is the parent 
    # directory of folder 'keyframes'
    # e.g. /home/maolin/workspace/mapping-data-polycam/garden/Sep1at10-00AM
    input_directory = "<your_input_directory>"

    # Your map name. Please note that the map name must consist of letters or numbers (A-Z/a-z/0-9), 
    # and must not contain spaces or special characters (such as -, _, /, etc.).
    map_name = "<your_map_name>"

    main(url, token, map_name, input_directory, visualize_poses=True, coordinate_frame_size=0.1, centroid_frame_size=0.5, split_groups=1, max_threads=4, use_first_image_pose_as_origo=False)

