from typing import List
import csv
import os
import json
import math
import numpy as np              # pip install numpy
from bs4 import BeautifulSoup   # pip install bs4 & pip install lxml


def parseSensors(soup: BeautifulSoup) -> List[dict]:
    """Finds all the different camera sensors in Metashape .xml export. Images can come from different sources and imaging sensors"""

    sensorsList = []
    sensors = soup.find_all("sensor")
    print(f'parsing sensors...')

    for sensor in sensors:
        sensor_id = int(sensor.attrs.get("id"))
        
        # Check if calibration node exists
        calibration = sensor.find("calibration")
        if calibration:
            sensor_type = calibration.get("type")
            resolution = calibration.find("resolution")
            width = int(resolution.attrs.get("width"))
            height = int(resolution.attrs.get("height"))

            f = float(calibration.find("f").contents[0]) if calibration.find("f") else 0.0
            cx = float(calibration.find("cx").contents[0]) if calibration.find("cx") else 0.0
            cy = float(calibration.find("cy").contents[0]) if calibration.find("cy") else 0.0
        else:
            sensor_type = sensor.get("type", "unknown")
            resolution = sensor.find("resolution")
            width = int(resolution.attrs.get("width")) if resolution else 0
            height = int(resolution.attrs.get("height")) if resolution else 0
            f, cx, cy = 0.0, 0.0, 0.0

        if sensor_type == 'spherical':
            data = {
                "sensor_type": sensor_type,
                "sensor_id": sensor_id,
                "width": width,
                "height": height,
                "f": 0.0,
                "ox": 0.0,
                "oy": 0.0,
            }
        else:
            data = {
                "sensor_type": sensor_type,
                "sensor_id": sensor_id,
                "width": width,
                "height": height,
                "f": f,
                "ox": width / 2 + cx,
                "oy": height / 2 + cy,
            }

        print(f'fx, fy: {data["f"]}\nox: {data["ox"]}\noy: {data["oy"]}\n')
        sensorsList.append(data)

    print(f'sensors parsed, sensorsList:\n{sensorsList}\n')
    return sensorsList


def parseComponents(soup: BeautifulSoup) -> List[dict]:
    """Finds all the components in Metashape .xml export. Each chunk can have many components"""

    componentsList = []
    components = soup.find_all("component")
    print(f'parsing components...')

    for component in components:
        component_id = int(component.attrs.get("id"))
        transform = component.find("transform")

        # create a default identity matrix for each component in case component doesn't have transform data
        transform_matrix = np.identity(4)

        if transform is not None:
            # if the component has been transformed in Metashape, replace identity matrix values with real transform data
            # if the component has not been translated, rotated, or scaled the specific values might be missing
            
            position = [0.0, 0.0, 0.0]
            rotation = np.identity(3).flatten()
            scale = 1.0

            try:
                position = transform.find("translation").contents[0].split(" ")
            except:
                print(f'warning: position in component {component_id} not found, defaulting to {position}')
            try:
                rotation = transform.find("rotation").contents[0].split(" ")
            except:
                print(f'warning: rotation in component {component_id} not found, defaulting to {rotation}')
            try:
                scale = transform.find("scale").contents[0]
            except:
                print(f'warning: scale in component {component_id} not found, defaulting to {scale}')

            transform_matrix[0][0] = float(rotation[0])
            transform_matrix[0][1] = float(rotation[1])
            transform_matrix[0][2] = float(rotation[2])
            transform_matrix[0][3] = float(position[0])
            transform_matrix[1][0] = float(rotation[3])
            transform_matrix[1][1] = float(rotation[4])
            transform_matrix[1][2] = float(rotation[5])
            transform_matrix[1][3] = float(position[1])
            transform_matrix[2][0] = float(rotation[6])
            transform_matrix[2][1] = float(rotation[7])
            transform_matrix[2][2] = float(rotation[8])
            transform_matrix[2][3] = float(position[2])
            transform_matrix[3][0] = 0.0
            transform_matrix[3][1] = 0.0
            transform_matrix[3][2] = 0.0
            transform_matrix[3][3] = 1.0 * float(scale)

        data = {
            "component_id": component_id,
            "tx": transform_matrix[0][3],
            "ty": transform_matrix[1][3],
            "tz": transform_matrix[2][3],
            "r00": transform_matrix[0][0],
            "r01": transform_matrix[0][1],
            "r02": transform_matrix[0][2],
            "r10": transform_matrix[1][0],
            "r11": transform_matrix[1][1],
            "r12": transform_matrix[1][2],
            "r20": transform_matrix[2][0],
            "r21": transform_matrix[2][1],
            "r22": transform_matrix[2][2],
            "xf": transform_matrix,
        }

        componentsList.append(data)

    print(f'components parsed, componentsList:\n{componentsList}\n')
    return componentsList


def main(xml_filepath: str, input_images_directory: str) -> None:
    with open(xml_filepath, "r") as xml_handle:

        soup = BeautifulSoup(xml_handle, "xml")

        sensorsList = parseSensors(soup)

        # components will contains the correct scale, translate and rotate transforms for the Object based on references
        componentsList = parseComponents(soup)

        #
        # parse individual camera data, requires sensor and component data
        #

        # bounding box for debugging
        bb_min = [math.inf, math.inf, math.inf]
        bb_max = [-math.inf, -math.inf, -math.inf]

        cameras = soup.find_all("camera")
        print(f'parsing cameras...')

        for camera in cameras:
            # find matching sensor and component data for current camera
            sensor_id = camera.attrs.get("sensor_id")
            component_id = camera.attrs.get("component_id")

            if sensor_id is not None and component_id is not None:
                # https://stackoverflow.com/questions/9014058/creating-a-python-list-comprehension-with-an-if-and-break
                # tries to get the first matching item in the sensorList and componentsList, None if no sensor or component data for the current camera
                sensor = next((item for item in sensorsList if item["sensor_id"] == int(sensor_id)), None)
                component = next((item for item in componentsList if item["component_id"] == int(component_id)), None)

                if sensor is not None and component is not None:
                    fx = fy = sensor["f"]
                    ox = sensor["ox"]
                    oy = sensor["oy"]

                    filename = camera.attrs.get("label")

                    #
                    # build the final transform matrix for each camera (image)
                    #

                    camera_transform = camera.find("transform").contents[0].split()
                    component_xf = component["xf"]

                    camera_xf = np.empty((4, 4))

                    # multiplying by component_xf[3][3] is multiplying the position by the component scale
                    camera_xf[0][0] = float(camera_transform[0])
                    camera_xf[0][1] = float(camera_transform[1])
                    camera_xf[0][2] = float(camera_transform[2])
                    camera_xf[0][3] = float(camera_transform[3]) * component_xf[3][3]
                    camera_xf[1][0] = float(camera_transform[4])
                    camera_xf[1][1] = float(camera_transform[5])
                    camera_xf[1][2] = float(camera_transform[6])
                    camera_xf[1][3] = float(camera_transform[7]) * component_xf[3][3]
                    camera_xf[2][0] = float(camera_transform[8])
                    camera_xf[2][1] = float(camera_transform[9])
                    camera_xf[2][2] = float(camera_transform[10])
                    camera_xf[2][3] = float(camera_transform[11]) * component_xf[3][3]
                    camera_xf[3][0] = float(camera_transform[12])
                    camera_xf[3][1] = float(camera_transform[13])
                    camera_xf[3][2] = float(camera_transform[14])
                    camera_xf[3][3] = float(camera_transform[15])

                    xf = np.matmul(component_xf, camera_xf)

                    with np.printoptions(precision=2, suppress=True):
                        print(f'xf:\n{xf}')


                    #
                    # store required data to a json file next to the input image - ready for Cloud Service submission
                    #

                    # bounding box for debugging
                    bb_min = [min(x, bb_min[i]) for i, x in enumerate(xf[0:3, 3])]
                    bb_max = [max(x, bb_max[i]) for i, x in enumerate(xf[0:3, 3])]

                    data = {
                        "img": filename,
                        "px": xf[0][3],
                        "py": xf[1][3],
                        "pz": xf[2][3],
                        "r00": xf[0][0],
                        "r01": xf[0][1],
                        "r02": xf[0][2],
                        "r10": xf[1][0],
                        "r11": xf[1][1],
                        "r12": xf[1][2],
                        "r20": xf[2][0],
                        "r21": xf[2][1],
                        "r22": xf[2][2],
                        "fx": fx,
                        "fy": fy,
                        "ox": ox,
                        "oy": oy,
                    }

                    json_path = os.path.join(input_images_directory, f"{os.path.splitext(filename)[0]}.json")
                    print(f'saving {json_path}')

                    # read from existing json file if possible
                    if os.path.exists(json_path):
                        with open(json_path, "r") as infile:
                            existing_data = json.load(infile)
                            for key in ["altitude", "longitude", "latitude"]:
                                if key in existing_data:
                                    data[key] = existing_data[key]

                    with open(json_path, "w") as outfile:
                        pretty_print = json.dumps(data, indent=4)
                        outfile.write(pretty_print)

        # bounding box for debugging
        with np.printoptions(precision=2, suppress=True):
            print(f'bb_min:\t{np.array(bb_min)}')
            print(f'bb_max:\t{np.array(bb_max)}')

    print('done')


if __name__ == "__main__":

    # Path of the camera pose XML, 
    # e.g. xml_filepath = "/Users/maolin/workspaces/mapping-360/cybergeo/hyd-metashape/1/camera.xml"
    xml_filepath = "path_of_camera_xml"
    
    # Path of image directory
    # e.g. input_images_directory = "Path of the directory of images"
    input_images_directory = "/Users/maolin/workspaces/mapping-360/cybergeo/hyd-metashape/1/frame"
    
    main(xml_filepath, input_images_directory)


