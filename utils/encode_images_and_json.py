import struct
import math
import os
import json
from natsort import natsorted   # pip install natsort
import numpy as np # pip install numpy

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


def i2s(i: int) -> str:
    if(i == 0):
        return '0'
    try:
        return struct.pack('>i', i).hex()
    except:
        return 0


def f2s(f: float) -> str:
    if(f == 0):
        return '0'
    try:
        return struct.pack('>f', f).hex()
    except:
        return 0


def d2s(d: float) -> str:
    if(d == 0):
        return '0'
    try:
        return struct.pack('>d', d).hex()
    except:
        return 0


def euler2Mat(heading: float, attitude: float, bank: float) -> np.ndarray:
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


def mat2Euler(m: np.ndarray) -> list[float]:
    if(m[2][0] > 0.9998):
        heading = np.arctan2(m[0][2], m[2][2])
        attitude = np.pi / 2.0
        bank = 0.0
        return heading, attitude, bank
    if(m[1][0] < -0.9998):
        heading = np.arctan2(m[0][2], m[2][2])
        attitude = -np.pi / 2.0
        bank = 0.0
        return heading, attitude, bank
    heading = np.arctan2(-m[2][0], m[0][0])
    attitude = np.arcsin(m[1][0])
    bank = np.arctan2(-m[1][2], m[1][1])

    return [heading, attitude, bank]

def encodeFilename(data: dict) -> str:

    run = i2s(data["run"])
    ih = f'{data["ih"]:x}'
    px = f2s(data["px"])
    py = f2s(data["py"])
    pz = f2s(data["pz"])
    fx = f2s(data["fx"])
    fy = f2s(data["fy"])
    ox = f2s(data["ox"])
    oy = f2s(data["oy"])
    rh = f2s(data["rh"])
    ra = f2s(data["ra"])
    rb = f2s(data["rb"])
    latitude_ll = d2s(data["latitude_ll"])
    longitude_ll = d2s(data["longitude_ll"])
    altitude_ll = d2s(data["altitude_ll"])

    return f"hxyz_{run}_{ih}_{px}_{py}_{pz}_{fx}_{fy}_{ox}_{oy}_{rh}_{ra}_{rb}_{latitude_ll}_{longitude_ll}_{altitude_ll}"


def process(input_directory: str) -> None:

    if not os.path.exists(input_directory):
        print("Invalid input directory")
        return

    json_files = []
    for file in natsorted(os.listdir(input_directory)):
        if file.endswith('.json'):
            json_files.append(os.path.join(input_directory, file))

    images_and_data = []

    for j in json_files:
        with open(j, 'r') as json_file:
            json_data = json.load(json_file)
            image_path = os.path.join(os.path.dirname(j), f"{json_data['imagePath']}")
            x = {'image': image_path, 'data': json_data}
            images_and_data.append(x)

    print(f"Found {len(images_and_data)} pairs")

    for pair in images_and_data:
        data = pair["data"]

        m = np.empty([3,3])
        m[0][0] = data['r00']
        m[0][1] = data['r01']
        m[0][2] = data['r02']
        m[1][0] = data['r10']
        m[1][1] = data['r11']
        m[1][2] = data['r12']
        m[2][0] = data['r20']
        m[2][1] = data['r21']
        m[2][2] = data['r22']

        euler = mat2Euler(m)

        data = {
            "run": data["run"],
            "ih": data["index"],
            "px": data["px"],
            "py": data["py"],
            "pz": data["pz"],
            "fx": data["fx"],
            "fy": data["fy"],
            "ox": data["ox"],
            "oy": data["oy"],
            "rh": euler[0],
            "ra": euler[1],
            "rb": euler[2],
            "latitude_ll": data["latitude"],
            "longitude_ll": data["longitude"],
            "altitude_ll": data["altitude"]
        }

        encoded_fn = encodeFilename(data)

        dir = os.path.dirname(pair["image"])
        img, ext = os.path.splitext(pair["image"])
        newname = f"{dir}{os.path.sep}{encoded_fn}{ext}"
        os.rename(pair["image"], newname)

if __name__ == '__main__':

    # finds all image/json pairs in input_dir and renames the images
    # with encoded data from the json

    input_dir = r"Directory_of_your_exported_images_and_poses"
    process(input_dir)
