from typing import List, Union
import os
import concurrent.futures
from natsort import natsorted # pip install natsort
import struct
import numpy as np # pip install numpy
import time
import json
import requests # pip install requests
import cv2 # pip install opencv-python
import math
import open3d as o3d # pip install open3d
import statistics 


def LocalizeImage(url: str, token: str, mapId: int, imagesAndPoses: List[dict], index: int) -> str:
    """Returns any errors or localization results as a JSON formatted string. Binary version of the endpoint for faster image uploads.

    fx, fy      Float values for the horizontal and vertical pixel focal length for the image
    ox, oy      Float values for the principal point offset for the image
    b64         Base64 encoded bytes containing the image as a .png file
    mapIds      A list of dictionaries of mapIds to localize against. A maximum of 8 maps can be used simultaneously

    success     Boolean for whether the image could be localized or not
    map         Integer of the localized map's id. When localizing to multiple maps at once, only one result is returned
    px, py, pz  Float values for the localized position
    r00-r22     Float values for a 3x3 matrix for the localized rotation
    time        Float value for the time it took for the server to localize
    """

    complete_url = url + "/localize"

    with open(imagesAndPoses[index]['image'], 'rb') as imgFile:
        img_bytes = imgFile.read()

        with open(imagesAndPoses[index]['pose'], 'r') as jsonFile:
            jsonData = json.load(jsonFile)

            data = {
                "token": token,
                "fx": jsonData['fx'],
                "fy": jsonData['fy'],
                "ox": jsonData['ox'],
                "oy": jsonData['oy'],
                "mapIds": [{"id": mapId}],
            }

            json_data = json.dumps(data)
            json_bytes = json_data.encode()

            body = json_bytes + b"\0" + img_bytes

            r = requests.post(complete_url, data=body)
            j = json.loads(r.text)

            j['imagePath'] = os.path.basename(imagesAndPoses[index]['image'])
            j['imageIndex'] = index
            j['request_data'] = data

            return json.dumps(j)

def saveResults(inputDirectory: str, mapId: int, results: List[dict]) -> str:

    output_directory = os.path.join(os.getcwd(), 'results')
    output_filepath = os.path.join(output_directory, f'{mapId}_results.json')

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    successful_times = [r["time"] for r in results if r["success"]]
    failed_times = [r["time"] for r in results if not r["success"]]

    data = {
        "from": inputDirectory,
        "against" : mapId,
        "successful" : len(successful_times),
        "total" : len(results),
        "ratio" : len(successful_times) / len(results),
        "mean_successful_time" : statistics.mean(successful_times) if len(successful_times) > 0 else None,
        "mean_failed_time" : statistics.mean(failed_times) if len(failed_times) > 0 else None,
        "results" : results,
    }

    with open(output_filepath, 'w') as new_file:
        json.dump(data, new_file, indent=4)

    return output_filepath


def DownloadSparse(url: str, token: str, mapId: int, save_file: bool=False) -> bytes:
    complete_url = url + "/sparse"

    data = {
        "token": token,
        "id": mapId,
    }

    r = requests.get(complete_url, params=data)
    b = r.content

    if(save_file):
        open(str(mapId) + '.ply', 'wb').write(b)

    return b


def project_to_screenspace(image_path: str, points: np.ndarray, camera_extrinsics: np.ndarray, camera_intrinsics: np.ndarray, half_res: bool=True) -> None:

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    inv = np.linalg.inv(camera_extrinsics)
    a = inv.dot(points).T[0][:, :3]
    b = camera_intrinsics.dot(a.T).T
    temp = np.tile(b[:, 2], (3, 1)).T
    c = b[:,:3] / temp
    c[:,2] = temp[:,2]

    #valid = (c[:,0] >= 0) & (c[:,0] < w) & \
    #        (c[:,1] >= 0) & (c[:,1] < h)

    valid = (c[:,0] >= 0) & (c[:,0] < w) & \
            (c[:,1] >= 0) & (c[:,1] < h) & \
            (c[:,2] >= 0)

    filtered_coordinates = c[valid]

    for point in filtered_coordinates:
        img = cv2.circle(img, (int(point[0]), int(point[1])), radius=5, color=(25, 255, 221), thickness=-1, lineType=cv2.LINE_AA)

    if half_res:
        img = cv2.resize(img, (math.floor(w/2), math.floor(h/2)), interpolation=cv2.INTER_AREA)

    cv2.imshow("points", img) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()



def visualizeResults(results_json_path: str, coordinate_frame_size: float=0.2, renderPoints: bool=False) -> None:

    with open(results_json_path, 'r') as json_file:
        data = json.load(json_file)

        root = data['from']

        mapId = data["against"]

        # hex 171A1F - Immersal dark blue
        background_color = [23, 26, 31]
        normalized_background_color = [c / 255.0 for c in background_color]

        # hex DDFF19 - Immersal yellow
        point_color = [221, 255, 25]
        normalized_point_color = [c / 255.0 for c in point_color]

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        opt = vis.get_render_option()
        opt.background_color = normalized_background_color
        opt.point_size = 2

        # sparse point cloud
        map_bytes = DownloadSparse(url, token, mapId, save_file=True)
        pcd = o3d.io.read_point_cloud(f'{mapId}.ply', format='ply')
        cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.5)
        vis.add_geometry(cl)

        points = np.asarray(pcd.points)
        points = np.hstack((points, np.ones((points.shape[0], 1), points.dtype)))
        points = points[:, :, np.newaxis]

        # localization results
        for r in data["results"]:
            image_path = os.path.join(root, r['imagePath'])

            if r['success']:
                mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coordinate_frame_size, origin=[0, 0, 0])
                T = np.array([   [r["r00"],  r["r01"],  r["r02"], r["px"]],
                                 [r["r10"],  r["r11"],  r["r12"], r["py"]],
                                 [r["r20"],  r["r21"],  r["r22"], r["pz"]],
                                 [0,         0,         0,        1     ]])
                mesh.transform(T)
                vis.add_geometry(mesh)

                if renderPoints:

                    data = r['request_data']

                    camera_intrinsics = np.array([[data['fx'],  0.0,         data['ox']],
                                                  [0.0,         data['fy'],  data['oy']],
                                                  [0.0,         0.0,         1.0       ]])


                    camera_extrinsics = np.array([[r['r00'],  r['r01'], r['r02'], r['px']],
                                                  [r['r10'],  r['r11'], r['r12'], r['py']],
                                                  [r['r20'],  r['r21'], r['r22'], r['pz']],
                                                  [0.0,       0.0,      0.0,      1.0    ]])

                    project_to_screenspace(image_path, points, camera_extrinsics, camera_intrinsics)


        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        # vis.add_geometry(origin)

        vis.run()
        vis.destroy_window()

def main(url: str, token: str, inputDirectory: str, mapId: id, maxThreads: int=1) -> None:
    """Collects image files and matching camera pose files as a list of dictionaries and tries to localize the images (multithreaded if maxThreads > 1) against a map in the Cloud Service"""

    imagesAndPoses = [{'image': os.path.join(inputDirectory, file), 'pose': os.path.join(inputDirectory, file.replace('.png', '.json'))} for file in natsorted(os.listdir(inputDirectory)) if file.endswith('.png')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=maxThreads) as executor:
        r = [executor.submit(LocalizeImage, url, token, mapId, imagesAndPoses, i) 
                   for i in range(len(imagesAndPoses))]
        results = []
        total = 0
        localized = 0
    
        for f in concurrent.futures.as_completed(r):
            result = json.loads(f.result())
            results.append(result)

            error = result['error']
            total += 1
            if error == 'none':
                success = result['success']
                print(f'localization in {result["time"]:0.2f}s: {success}\tpx: {result["px"]:.03f}\tpy: {result["py"]:.03f}\tpz: {result["pz"]:.03f}')
                if success:
                    localized += 1
            else:
                print(f'error: {error}')

        print(f'successfully localized: {localized}/{total} = {localized / total:.4f}')

        if len(results) == 0:
            print("no results, aborting")
            return

        results_json_path = saveResults(inputDirectory, mapId, results)
        visualizeResults(results_json_path, coordinate_frame_size=0.3, renderPoints=True)

if __name__ == '__main__': 
    url = 'https://api.immersal.com'
    token = 'your-token-here'
    inputDirectory = r'.\input_images_and_poses'
    mapId = 12345

    main(url, token, inputDirectory, mapId)