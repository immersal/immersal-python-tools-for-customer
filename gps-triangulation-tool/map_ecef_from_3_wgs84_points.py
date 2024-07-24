import numpy as np
import math
import json
import requests # pip install requests


def API_metadataset(url: str, token: str, data: dict, map_id: int) -> str:
    complete_url = url + "/metadataset"

    data["token"] = token
    data["id"] = map_id

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def normalize_vector(v: np.ndarray) -> np.ndarray:
    ''' returns a normalized vector of v '''
    vector_length = np.linalg.norm(v)
    if vector_length == 0:
        return [0, 0, 0]
    return v / vector_length


def wgs84_to_ECEF(wgs84: np.ndarray) -> np.ndarray:
    lat = wgs84[0]
    lon = wgs84[1]
    alt = wgs84[2]

    rad_lat = np.double(lat * (math.pi / 180.0))
    rad_lon = np.double(lon * (math.pi / 180.0))

    a = np.double(6378137.0)
    finv = np.double(298.2572235604902)
    f = 1 / finv
    e2 = 1 - (1 - f) * (1 - f)
    v = a / math.sqrt(1 - e2 * math.sin(rad_lat) * math.sin(rad_lat))

    x = (v + alt) * math.cos(rad_lat) * math.cos(rad_lon)
    y = (v + alt) * math.cos(rad_lat) * math.sin(rad_lon)
    z = (v * (1 - e2) + alt) * math.sin(rad_lat)

    return np.array([x, y, z])


def transform_matrix4x4_from_3_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab = normalize_vector(b - a)
    ac = normalize_vector(c - a)

    up = np.cross(ac, ab)
    cross = np.cross(ab, up)

    m = np.identity(4)

    # set rotation x component
    m[:3, 0] = ab

    # set rotation y component
    m[:3, 1] = up

    # set rotation z component
    m[:3, 2] = cross

    # set position component
    m[:3, 3] = a

    return m


def matrix_to_quaternion(m: np.ndarray) -> np.ndarray:
    qw = 1.0
    qx = 0.0
    qy = 0.0
    qz = 0.0

    tr = m[0][0] + m[1][1] + m[2][2]

    if (tr > 0): 
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S;
        qx = (m[2][1] - m[1][2]) / S
        qy = (m[0][2] - m[2][0]) / S 
        qz = (m[1][0] - m[0][1]) / S
    elif (m[0][0] > m[1][1]) and (m[0][0] > m[2][2]):
        S = np.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        qw = (m[2][1] - m[1][2]) / S
        qx = 0.25 * S
        qy = (m[0][1] + m[1][0]) / S
        qz = (m[0][2] + m[2][0]) / S
    elif (m[1][1] > m[2][2]):
        S = np.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        qw = (m[0][2] - m[2][0]) / S
        qx = (m[0][1] + m[1][0]) / S
        qy = 0.25 * S
        qz = (m[1][2] + m[2][1]) / S
    else: 
        S = np.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        qw = (m[1][0] - m[0][1]) / S
        qx = (m[0][2] + m[2][0]) / S
        qy = (m[1][2] + m[2][1]) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])


def main() -> None:

    # 3 manually picked points in the map x, y, z
    map_position_a = np.array([0, 0, 0])
    map_position_b = np.array([0, 0,  0])
    map_position_c = np.array([0, 0,  0 ])

    # latitude, longitude, and altitude values for the 3 points
    wgs84_a = np.array([0.0, 0.0, 0.0])
    wgs84_b = np.array([0.0, 0.0, 0.0])
    wgs84_c = np.array([0.0, 0.0, 0.0])

    # ecef coordinates for the 3 points
    ecef_a = wgs84_to_ECEF(wgs84_a)
    ecef_b = wgs84_to_ECEF(wgs84_b)
    ecef_c = wgs84_to_ECEF(wgs84_c)


    # build transform matrices from the 3 points, these describe the orientation and position in relation to point a and direction to b and c
    map_matrix = transform_matrix4x4_from_3_points(map_position_a, map_position_b, map_position_c)
    ecef_matrix = transform_matrix4x4_from_3_points(ecef_a, ecef_b, ecef_c)

    # compute scale difference, not used here though but it should be 1.0 if the map is in correct scale
    # just a difference in length between points a and b in both map and ecef space
    scale = np.linalg.norm(ecef_b - ecef_a) / np.linalg.norm(map_position_b - map_position_a)

    # compute the matrix that transforms map_matrix to ecef_matrix
    xf_matrix = np.matmul(ecef_matrix, np.linalg.inv(map_matrix))

    # convert the rotation part of the 4x4 matrix into a quaternion for our REST API
    q = matrix_to_quaternion(xf_matrix)

    # extract the position part of the 4x4 matrix into a quaternion for our REST API
    pos = xf_matrix[:3, 3]

    # use REST API to update map metadata
    url = 'https://developers.immersal.com/'
    token = 'your token here'
    map_id = 11111
    data = {
        "tx": pos[0],
        "ty": pos[1],
        "tz": pos[2],
        "qw": q[0],
        "qx": q[1],
        "qy": q[2],
        "qz": q[3],
    }
    print(API_metadataset(url, token, data, map_id))


    # DEBUG
    with np.printoptions(precision=5, suppress=True):
        # print(map_matrix)
        # print(ecef_matrix)
        print(scale)
        print(xf_matrix)
        # print(q)
        # print(pos)
        print(json.dumps(data, indent=4))


if __name__ == '__main__':
    main()
