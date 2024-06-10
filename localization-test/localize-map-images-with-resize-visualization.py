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


def s2i(s: str) -> int:
    """Decodes a string value from the filename to an integer value"""

    try:
        f = s.zfill(8)
        return struct.unpack('>i', bytes.fromhex(f))[0]
    except:
        return 0


def s2f(s: str) -> float:
    """Decodes a string value from the filename to a float value"""

    try:
        return struct.unpack('>f', bytes.fromhex(s))[0]
    except:
        return 0


def s2d(s: str) -> float:
    """Decodes a string value from the filename to a double value"""

    try:
        return struct.unpack('>d', bytes.fromhex(s))[0]
    except:
        return 0


def Euler2Mat(heading: float, attitude: float, bank: float) -> np.ndarray:
    """Converts heading, attitude, and bank rotation values into a 3x3 rotation matrix"""

    m = np.empty([3,3])
    ch = np.cos(np.double(heading))
    sh = np.sin(np.double(heading))
    ca = np.cos(np.double(attitude))
    sa = np.sin(np.double(attitude))
    cb = np.cos(np.double(bank))
    sb = np.sin(np.double(bank))

    m[0][0] = ch * ca
    m[0][1] = sh * sb - ch * sa * cb
    m[0][2] = ch * sa * sb + sh * cb
    m[1][0] = sa
    m[1][1] = ca * cb
    m[1][2] = -ca * sb
    m[2][0] = -sh * ca
    m[2][1] = sh * sa * cb + ch * sb
    m[2][2] = -sh * sa * sb + ch * cb

    return m


def ParseFilenameToImageData(filename: str) -> dict:
    """ Parses a filename from an image downloaded from the Cloud Service to image metadata such as the camera pose

    hxyz_[run]_[index]_[px]_[py]_[pz]_[fx]_[fy]_[ox]_[oy]_[rh]_[ra]_[rb]_[latitude]_[longitude]_[altitude]_[hash?].png

    run         Integer for the tracker session. increment if tracking is lost or image is from a different session
    index       Running integer index for the images in the same run
    px, py, pz  Float values for the image's camera position. Used to always set metric scale and also as constraints if preservePoses is True in a REST API construct request
    rh, ra, rb  Float values for image's camera orientation as heading, attitude, and bank
    fx, fy      Float values for the horizontal and vertical pixel focal length for the image
    ox, oy      Float values for the principal point offset for the image
    latitude    Double value for the WGS84 latitude for the image's camera position (optional)
    longitude   Double value for the WGS84 longitude for the image's camera position (optional)
    altitude    Double value for the WGS84 altitude for the image's camera position (optional)

    anchor image is marked with a run and index value of 0. Only one can exist in the map input images
    """

    components = os.path.splitext(os.path.basename(filename))[0].split('_')

    rh = s2f(components[10])
    ra = s2f(components[11])
    rb = s2f(components[12])
    m = Euler2Mat(rh, ra, rb)

    run = s2i(components[1]),   
    index = int(components[2], 16)
    isAnchor = run == 0 and index == 0

    imageData = {
        "run" : s2i(components[1]),
        "index" : int(components[2], 16),
        "anchor" : isAnchor,
        "px" : s2f(components[3]),
        "py" : s2f(components[4]),
        "pz" : s2f(components[5]),
        "fx" : s2f(components[6]),
        "fy" : s2f(components[7]),
        "ox" : s2f(components[8]),
        "oy" : s2f(components[9]),
        "m" : m,
        "latitude" : s2d(components[13]),
        "longitude" : s2d(components[14]),
        "altitude" : s2d(components[15]),
    }

    return imageData


def LocalizeImage(url: str, token: str, mapId: int, imagesList: List[str], index: int, resize_factor: float=1.0) -> str:
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

    with open(imagesList[index], 'rb') as imgFile:
        img_bytes = imgFile.read()
        imageData = ParseFilenameToImageData(imagesList[index])

        if(resize_factor != 1.0):
            nparray = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparray, cv2.IMREAD_GRAYSCALE)

            height, width = img.shape

            new_height = math.floor(height * resize_factor)
            new_width = math.floor(width * resize_factor)

            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            img_bytes = cv2.imencode('.png', resized_img)[1].tobytes()

            # cv2.imshow('image', resized_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        data = {
            "token": token,
            "fx": imageData['fx'] * resize_factor,
            "fy": imageData['fy'] * resize_factor,
            "ox": imageData['ox'] * resize_factor,
            "oy": imageData['oy'] * resize_factor,
            "mapIds": [{"id": mapId}],
        }

        json_data = json.dumps(data)
        json_bytes = json_data.encode()

        body = json_bytes + b"\0" + img_bytes

        r = requests.post(complete_url, data=body)
        j = json.loads(r.text)

        j['imagePath'] = os.path.basename(imagesList[index])
        j['imageIndex'] = index
        j['request_data'] = data
        
        return j


def saveResults(inputDirectory: str, mapId: int, resize_factor: float, results: List[dict]) -> str:

    output_directory = os.path.join(os.getcwd(), 'results')
    output_filepath = os.path.join(output_directory, f'{mapId}_results.json')

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    successful_times = [r["time"] for r in results if r["success"]]
    failed_times = [r["time"] for r in results if not r["success"]]

    data = {
        "from": inputDirectory,
        "against" : mapId,
        "resize_factor" : resize_factor,
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


def main(url: str, token: str, inputDirectory: str, mapId: id, resize_factor: float=1.0, maxThreads: int=1, renderPoints: bool=False) -> None:
    """Collects image files, parses image metadata from filenames and tries to localize the images (multithreaded) against a map in the Cloud Service"""

    imagesList = [os.path.join(inputDirectory, file) for file in natsorted(os.listdir(inputDirectory)) if file.endswith('.png')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=maxThreads) as executor:
        r = [executor.submit(LocalizeImage, url, token, mapId, imagesList, i, resize_factor) for i in range(0, len(imagesList))]
        results = []
        total = 0
        localized = 0
    
        for f in concurrent.futures.as_completed(r):
            r = f.result()
            results.append(r)

            error = r['error']
            total += 1
            if error == 'none':
                success = r['success']
                print(f'localization in {r["time"]:0.2f}s: {success}\tpx: {r["px"]:.03f}\tpy: {r["py"]:.03f}\tpz: {r["pz"]:.03f}')
                if success:
                    localized += 1
            else:
                print(f'error: {error}')

        # print(f'successfully localized: {localized}/{total}')
        print(f'successfully localized: {localized}/{total} = {localized / total:.4f}')

        if len(results) == 0:
            print("no results, aborting")
            return

        results_json_path = saveResults(inputDirectory, mapId, resize_factor, results)
        # visualizeResults(results_json_path, coordinate_frame_size=0.3, renderPoints=True)


if __name__ == '__main__':

    # Immersal international server
    url = 'https://api.immersal.com'
    token = "your_token"

    # Immersal China server
    # url = 'https://immersal.hexagon.com.cn'
    # token = "your_token"

    # Path of your images
    # e.g. inputDirectory = r'/Users/maolin/workspaces/im-maps/tencent/4229/img'
    inputDirectory = "path_of_frames"

    # Your map ID, which should be a number
    mapId = 123

    main(url, token, inputDirectory, mapId, resize_factor=1.0, maxThreads=4, renderPoints=True)