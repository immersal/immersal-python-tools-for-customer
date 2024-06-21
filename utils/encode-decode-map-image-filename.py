import struct
import math
import os
import json
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


def decodeFilename(filename: str) -> dict:
    '''
    run = identifier for the current session. images in the same coordinate system, same tracking session should have the same run value
    ih = running integer for images
    px = position x
    py = position y
    pz = position z
    fx = horizontal pixel focal length
    fy = vertical pixel focal length
    ox = horizontal principal point offset
    oy = vertical principal point offset
    rh = rotation, heading
    ra = rotation, attitude
    rb = rotation, bank
    latitude = WGS84 latitude
    longitude = WGS84 longitude
    altitude = WGS84 altitude
    '''

    components = os.path.splitext(filename)[0].split("_")

    run = s2i(components[1])
    ih = int(components[2], 16)
    px = s2f(components[3])
    py = s2f(components[4])
    pz = s2f(components[5])
    fx = s2f(components[6])
    fy = s2f(components[7])
    ox = s2f(components[8])
    oy = s2f(components[9])
    rh = s2f(components[10])
    ra = s2f(components[11])
    rb = s2f(components[12])
    latitude = s2d(components[13])
    longitude = s2d(components[14])
    altitude = s2d(components[15])

    data = {
        "run" : run,
        "ih" : ih,
        "px" : px,
        "py" : py,
        "pz" : pz,
        "fx" : fx,
        "fy" : fy,
        "ox" : ox,
        "oy" : oy,
        "rh" : rh,
        "ra" : ra,
        "rb" : rb,
        "latitude_ll" : latitude,
        "longitude_ll" : longitude,
        "altitude_ll" : altitude,
    }

    return data


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

    return f"hxyz_{run}_{ih}_{px}_{py}_{pz}_{fx}_{fy}_{ox}_{oy}_{rh}_{ra}_{rb}_{latitude_ll}_{longitude_ll}_{altitude_ll}.png"


def do_decode(source_filename: str) -> None:

    # decode filename into a dictionary of values
    decoded_values = decodeFilename(source_filename)
    print(f"decoded values as pretty printed .json:\n{json.dumps(decoded_values, indent=4)}")


    # convert euler angles to 3x3 rotation matrix and back
    with np.printoptions(precision=3, suppress=True):
        m = euler2Mat(decoded_values["rh"], decoded_values["ra"], decoded_values["rb"])
        print(f"\nrotation as 3x3 matrix:\n{m}")
        e = mat2Euler(m)
        print(f"rotation converted back to euler angles:\n{e}")

    #
    # encode dictionary values back to filename, notice the Cloud Service adds a unique identifier when submitting an image https://man7.org/linux/man-pages/man3/mkostemps.3.html
    # that last component is not relevant metadata and be discarded at least for most purposes
    #
    encoded_filename = encodeFilename(decoded_values)
    print(f"\noriginal:\t{source_filename}")
    print(f"encoded:\t{encoded_filename}")


def do_encode(data: dict) -> None:
    filename = encodeFilename(data)
    print(f"encoded:\t{filename}")


if __name__ == '__main__':
    
    # Decoding
    # source_filename = r"hxyz_0_1e_be114320_3e2b84c0_bea72966_44b6cff7_44b6cff7_4469fc02_44343e8f_3fb9b3b7_bfac0e01_3fd40caf_403ead9d9ad02f0f_405a0660bb03d173_407f3c5c20000000_nUx88b.png"
    # do_decode(source_filename)

    # Encoding
    data = {
        "run": 1,
        "ih": 1,
        "px": 0,
        "py": 0,
        "pz": 0,
        "fx": 1539.27064,
        "fy": 539.27064,
        "ox": 960,
        "oy": 500.5,
        "rh": 0,
        "ra": 0,
        "rb": 0,
        "latitude_ll": 0,
        "longitude_ll": 0,
        "altitude_ll": 0
    }
    do_encode(data)
