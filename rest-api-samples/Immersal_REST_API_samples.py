from typing import List
import requests             # pip install requests
import json
import base64


def Login(url: str, username: str, password: str) -> str:
    """Returns any errors or the user's userId, token, and level as a JSON formatted string."""

    complete_url = url + "/login"

    data = {
        "login": username,
        "password": password,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def Version(url: str) -> str:
    """Returns any errors or the user's userId, token, and level as a JSON formatted string."""

    complete_url = url + "/version"

    r = requests.get(complete_url)

    return r.text


def AccountStatus(url: str, token: str) -> str:
    """Returns any errors or the user's userId, imageCount, imageMax, eulaAccepted as a JSON formatted string.

    imageCount is the amount of images in the user's workspace.
    imageMax is the maximum allowed images for the user's workspace.
    """

    complete_url = url + "/status"

    data = {
        "token": token,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def SetMapAccessToken(url: str, token: str, mapId: int) -> str:
    """Returns any errors or the mapId and accessToken for the map as a JSON formatted string.
    Each request to this endpoint will reset and generate a new accessToken for the map.

    accessToken can be used to download and localize to the map without the Developer Token
    """

    complete_url = url + "/setmaptoken"

    data = {
        "token": token,
        "id": mapId,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def ClearMapAccessToken(url: str, token: str, mapId: int) -> str:
    """Returns any errors or the mapId and accessToken for the map as a JSON formatted string.

    accessToken will be set to 'cleared' removing all access without a valid Developer Token.
    """

    complete_url = url + "/clearmaptoken"

    data = {
        "token": token,
        "id": mapId,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


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
    return r.text


def RestoreMap(url: str, token: str, mapId: int, clearWorkspace: bool=True) -> str:
    """Returns any errors for the request. Restores existing map data to the user's workspace.

    If clear is set to False, existin images in the workspace are kept and restored map images are added to the workspace.
    When set to True, the whole workspace is cleared of existing images and overwritten with the restored map's images.
    """

    complete_url = url + "/restore"

    data = {
        "token": token,
        "id": mapId,
        "clear": clearWorkspace,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data, verify=False)
    return r.text


def EditMap(url: str, token: str, mapId: int, mapName: str, editPath: str, radius: float = 0.001) -> str:
    """
    """
    complete_url = url + "/editmap"

    with open(editPath, 'rb') as mapFile:
        ply_bytes = mapFile.read()

        data = {
            "token": token,
            "name": mapName,
            "radius": radius,
            "id": mapId,
        }

        json_data = json.dumps(data)
        json_bytes = json_data.encode()

        body = json_bytes + b"\0" + ply_bytes

        r = requests.post(complete_url, data=body)
        return r.text


def ListJobs(url: str, token: str) -> str:
    """Returns any errors or all of the user's maps and their metadata as a JSON formatted string."""

    complete_url = url + "/list"

    data = {
        "token": token,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def ListJobsGPS(url: str, token: str, latitude: float, longitude: float, radius :float) -> str:
    """Returns any errors or the user's maps and their metadata within a radius from a given lat/long coordinate as a JSON formatted string."""

    complete_url = url + "/geolist"

    data = {
        "token": token,
        "latitude": latitude,
        "longitude": longitude,
        "radius": radius,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def ConvertToBase64(src_filepath: str) -> bytes:
    """Returns base64 encoded bytes of a given file. Used to encode images for on-server localization."""

    with open(src_filepath, "rb") as imageFileAsBinary:
        fileContent = imageFileAsBinary.read()
        b64_encoded_img = base64.b64encode(fileContent)

        return b64_encoded_img


def SubmitImageB64(url: str, token: str, imagePath: str) -> str:
    """Returns any errors or the path of the submitted image in the Cloud Service as a JSON formatted string. Submits an image to the user's workspace.

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
    b64         Base64 encoded bytes containing the image as a .png file
    """

    complete_url = url + "/captureb64"

    data = {
        "token": token,
        "run": 0,
        "index": 0,
        "anchor": False,
        "px": 0.0,
        "py": 0.0,
        "pz": 0.0,
        "r00": 1.0,
        "r01": 0.0,
        "r02": 0.0,
        "r10": 0.0,
        "r11": 1.0,
        "r12": 0.0,
        "r20": 0.0,
        "r21": 0.0,
        "r22": 1.0,
        "fx": 0.0,
        "fy": 0.0,
        "ox": 0.0,
        "oy": 0.0,
        "latitude" : 0.0,
        "longitude" : 0.0,
        "altitude" : 0.0,
        "b64": str(ConvertToBase64(imagePath), "utf-8"),
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def SubmitImage(url: str, token: str, imagePath: str) -> str:
    """Returns any errors or the path of the submitted image in the Cloud Service as a JSON formatted string. Submits an image to the user's workspace. Binary version of the endpoint for faster image uploads.

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

    with open(imagePath, 'rb') as imgFile:
        img_bytes = imgFile.read()

        data = {
            "token": token,
            "run": 0,
            "index": 0,
            "anchor": False,
            "px": 0.0,
            "py": 0.0,
            "pz": 0.0,
            "r00": 1.0,
            "r01": 0.0,
            "r02": 0.0,
            "r10": 0.0,
            "r11": 1.0,
            "r12": 0.0,
            "r20": 0.0,
            "r21": 0.0,
            "r22": 1.0,
            "fx": 0.0,
            "fy": 0.0,
            "ox": 0.0,
            "oy": 0.0,
            "latitude" : 0.0,
            "longitude" : 0.0,
            "altitude" : 0.0,
        }

        json_data = json.dumps(data)
        json_bytes = json_data.encode()

        body = json_bytes + b"\0" + img_bytes

        r = requests.post(complete_url, data=body)
        return r.text


def SubmitB2G(url: str, token: str, b2gPath: str) -> str:
    """Returns any errors or the path of the submitted image in the Cloud Service as a JSON formatted string. Submits a b2g file to the user's workspace.

    Below sample modifies the rotation matrix and converts the BLK2GO right-handed Z-up coordinate system to Immersal's Y-up

    px, py, pz  Float values for the image's camera position. Used to always set metric scale and also as constraints if preservePoses is True in a REST API construct request
    r00-r22     Float values for a 3x3 matrix of the image's camera rotation
    b2g_bytes   Bytes containing the .b2g file
    """

    complete_url = url + "/b2g"

    with open(b2gPath, 'rb') as b2gFile:
        b2g_bytes = b2gFile.read()

        data = {
            "token": token,
            "px": 0.0,
            "py": 0.0,
            "pz": 0.0,
            "r00": 1.0,
            "r01": 0.0,
            "r02": 0.0,
            "r10": 0.0,
            "r11": 0.0,
            "r12": 1.0,
            "r20": 0.0,
            "r21": -1.0,
            "r22": 0.0,
        }

        json_data = json.dumps(data)
        json_bytes = json_data.encode()

        body = json_bytes + b"\0" + b2g_bytes

        r = requests.post(complete_url, data=body)
        return r.text        


def StartMapConstruction(url: str, token: str, mapName: str) -> str:
    """Returns any errors or the new map's id and size (number of images) on the Cloud Service as a JSON formatted string. Submits the map construction job for the images in the user's workspace.
    
    featureCount    Integer for the max amount of features per image. Increases the total amount of features in the map but also increases map filesize
    preservePoses   Boolean to enable constraints input images' camera pose data as constraints. Speeds up map construction if input pose data is accurate
    """

    complete_url = url + "/construct"

    # data = {
    #     "token": token,
    #     "name": mapName,
    #     "featureCount": 1024,
    #     "preservePoses": True,
    # }

    data = {
        "token": token,
        "name": mapName,
        "preservePoses": False,
        # "featureCount": 1024,
        "featureCountMax": 262144,
        "featureFilter": 1,
        "trackLengthMin": 2,
        "priorBA": 0,
        "dense": 1,
        # "nnFilter": 24,
        # "featureType": 6,
        # "rescale": 960
        "triangulationDistanceMax": 64
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data, verify=False)
    return r.text


def SetSharingMode(url: str, token: str, mapId: int, privacy: int) -> str:
    """Returns any errors or the path of the submitted image in the Cloud Service as a JSON formatted string. privacy of 0 means private, 1 means public."""

    complete_url = url + "/privacy"

    data = {
        "token": token,
        "id": mapId,
        "privacy": privacy,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def ServerLocalizeB64(url: str, token: str, imagePath: str, mapId: int) -> str:
    """Returns any errors or localization results as a JSON formatted string.

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

    complete_url = url + "/localizeb64"

    data = {
        "token": token,
        "fx": 0.0,
        "fy": 0.0,
        "ox": 0.0,
        "oy": 0.0,
        "b64": str(ConvertToBase64(imagePath), "utf-8"),
        "mapIds": [{"id": mapId}],
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def ServerLocalize(url: str, token: str, imagePath: str, mapId: int) -> str:
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

    with open(imagePath, "rb") as fb:
        b = fb.read()

        data = {
            "token": token,
            "fx": 759.615,
            "fy": 759.615,
            "ox": 251.492,
            "oy": 480.324,
            "mapIds": [{"id": mapId}],
        }

        json_data = json.dumps(data)
        json_bytes = json_data.encode()

        body = json_bytes + b"\0" + b

        r = requests.post(complete_url, data=body)
        return r.text


def ServerLocalizeGeoPose(url: str, token: str, imagePath: str, mapId: int) -> str:
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

    complete_url = url + "/geopose"

    with open(imagePath, "rb") as fb:
        b = fb.read()

        data = {
            "token": token,
            "fx": 1431.3203125,
            "fy": 1431.3203125,
            "ox": 966.492,
            "oy": 720.324,
            "mapIds": [{"id": mapId}],
        }

        json_data = json.dumps(data)
        json_bytes = json_data.encode()

        body = json_bytes + b"\0" + b

        r = requests.post(complete_url, data=body)
        return r.text


def DownloadMap(url: str, token: str, mapId: int) -> str:
    """Returns any errors or base64 encoded .bytes map file used for localization and it's sha256 hash as a JSON formatted string."""

    complete_url = url + "/sparse"

    data = {
        "token": token,
        "id": mapId,
    }

    r = requests.get(complete_url, params=data)
    map_bytes = r.content
    
    # write file on disk
    # open(f"{mapId}.ply", 'wb').write(map_bytes)

    return r.text


def GetCoverage(url: str, token: str, mapId: int) -> str:
    """Returns any errors or map id and component info for stitch and align jobs as a JSON formatted string.

    connected       map ids for the components that could be stitched or aligned
    disconnected    map ids for the components that could not be stitched or aligned. Likely due to not enough overlap.
    """

    complete_url = url + "/coverage"

    data = {
        "token": token,
        "id": mapId,
    }

    r = requests.get(complete_url, params=data)
    return r.text


def StitchMaps(url: str, token: str, mapName: str, mapIds: List[int]) -> str:
    """Returns any errors or the new map's id and size (number of input maps) on the Cloud Service as a JSON formatted string. Starts the stitching job for the input maps which fuses them together."""

    complete_url = url + "/fuse"

    data = {
        "token": token,
        "name": mapName,
        "mapIds": mapIds,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def AlignMaps(url: str, token: str, jobName: str, mapIds: List[int]) -> str:
    """Returns any errors or the align job's id and size (number of input maps) on the Cloud Service as a JSON formatted string. Starts the alignment job for the input maps. Only modifies input map metadata, does not create a new map"""

    complete_url = url + "/align"

    data = {
        "token": token,
        "name": jobName,
        "mapIds": mapIds,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def CopyMaps(url: str, token: str, login: str, mapId: int) -> str:
    """Returns any errors as a JSON formatted string. Copies the map to another user. The copy will have a new map id.

    login   String value of the recipient's user id
    """

    complete_url = url + "/copy"

    data = {
        "token": token,
        "login": login,
        "id": mapId,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def ResetPassword(url: str, token: str, newPassword: str) -> str:
    """Returns any errors as a JSON formatted string. Resets the user's password"""

    complete_url = url + "/password"

    data = {
        "token": token,
        "password": newPassword,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def MetadataGet(url: str, token: str, mapId: int) -> str:
    """Returns any errors or map metadata as a JSON formatted string.

    tx, ty, tz, qw, qx, qy, qz, and scale are by default the map's coordinates in the ECEF coordinate system. These coordinates are derived from the map's input images' WGS84 lat/long/alt coordinates when available.
    The values can be set to represent the map's transform in a user-defined coordinate system and can also be reset to original ECEF coordinates.

    id              Integer for the map/job id
    type            Integer for the type of job in the Cloud Service. 0 for map construction, 1 for a map stitching job
    created         String for the job creation date
    version         String the the Cloud Service version
    user            Integer for the map's owner/user
    creator         Integer for the map's original creator. Different from the user if map was sent/copied to the user
    name            String for the map/job name
    size            Integer for the size of the job. Number of images if a map construction job, number of input maps if stitching or alignment job
    status          String for the status of the job. Pending, Processing, Sparse, Done, Failed. Status is Sparse when the map has been generated and the .bytes file is available but mesh output is still processed
    errno           Integer for any errors for the job. 0 if no errors
    privacy         Integer for the privacy level. 0 means private, 1 means public
    latitude        Float value for the map's WGS84 latitude
    longitude       Float value for the map's WGS84 longitude
    altitude        Float value for the map's WGS84 altitude
    tx              Float value for the map's ECEF coordinate x position
    ty              Float value for the map's ECEF coordinate y position
    tz              Float value for the map's ECEF coordinate z position
    qw              Float value for the map's ECEF coordinate rotation as a quaternion's w value
    qx              Float value for the map's ECEF coordinate rotation as a quaternion's x value
    qy              Float value for the map's ECEF coordinate rotation as a quaternion's y value
    qz              Float value for the map's ECEF coordinate rotation as a quaternion's z value
    scale           Float value for a scale multiplier for the map's ECEF coordinates
    sha256_al       String for the map's .bytes file's sha256
    sha256_sparse   String for the map's .ply point cloud file's sha256
    sha256_dense    String for the map's .ply vertex-colored mesh file's sha256        
    sha256_tex      String for the map's .glb textured mesh file's sha256
    """

    complete_url = url + "/metadataget"

    data = {
        "token": token,
        "id": mapId,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def MetadataSet(url: str, token: str,  mapId: int) -> str:
    """Returns any errors as a JSON formatted string. Sets the values for the available and specified map metadata.

    tx, ty, tz, qw, qx, qy, qz, and scale are by default the map's coordinates in the ECEF coordinate system. These coordinates are derived from the map's input images' WGS84 lat/long/alt coordinates when available.
    The values can be set to represent the map's transform in a user-defined coordinate system and can also be reset to original ECEF coordinates.

    name            String for the map/job name
    privacy         Integer for the privacy level. 0 means private, 1 means public
    latitude        Float value for the map's WGS84 latitude
    longitude       Float value for the map's WGS84 longitude
    altitude        Float value for the map's WGS84 altitude
    tx              Float value for the map's ECEF coordinate x position
    ty              Float value for the map's ECEF coordinate y position
    tz              Float value for the map's ECEF coordinate z position
    qw              Float value for the map's ECEF coordinate rotation as a quaternion's w value
    qx              Float value for the map's ECEF coordinate rotation as a quaternion's x value
    qy              Float value for the map's ECEF coordinate rotation as a quaternion's y value
    qz              Float value for the map's ECEF coordinate rotation as a quaternion's z value
    scale           Float value for a scale multiplier for the map's ECEF coordinates
    """

    complete_url = url + "/metadataset"

    data = {
        "token": token,
        "id": mapId,
        # "name" : 'newMapName',
        # "privacy" : 0,
        # "latitude" : 60.0,
        # "longitude" : 25.0,
        # "altitude" : 0.0,
        # "tx": 0.0,
        # "ty": 0.0,
        # "tz": 0.0,
        # "qw": 1.0,
        # "qx": 0.0,
        # "qy": 0.0,
        # "qz": 0.0,
        # "scale": 1.0,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def ResetAlignment(url: str, token: str, mapId: int) -> str:
    """Returns any errors as a JSON formatted string. Resets the map's transform to the original ECEF coordinates."""

    complete_url = url + "/reset"

    data = {
        "token": token,
        "id": mapId,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


def DeleteMap(url: str, token: str, mapId: int) -> str:
    """Returns any errors as a JSON formatted string. Deletes the map from the user. Can not be undone. Does not delete any previously sent copies of the map"""

    complete_url = url + "/delete"

    data = {
        "token": token,
        "id": mapId,
    }

    json_data = json.dumps(data)

    r = requests.post(complete_url, data=json_data)
    return r.text


if __name__ == '__main__':

    # Immersal international server
    url = 'https://api.immersal.com'
    token = "your_token"

    # Immersal China server
    # url = 'https://immersal.hexagon.com.cn'
    # token = "your_token"

    # do your work here.



