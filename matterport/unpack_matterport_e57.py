from typing import List
import os
import json
import numpy as np              # pip install numpy
import pye57                    # pip install pye57

# note: pye57 requires manual wheel compilation and installation on macos
# see e.g. https://jakubuhlik.com/docs/pcv/docs.html#libraries-building

# Matterport image order
# 0 = UP
# 1 = FRONT
# 2 = RIGHT
# 3 = BACK
# 4 = LEFT
# 5 = DOWN

class MatterportImage:
    def __init__(self, node: pye57.libe57.StructureNode):
        self.node = node
        self.guid = node["guid"].value()
        self.name = node["name"].value()

        # representation
        self.repString = ""
        if (node.isDefined("visualReferenceRepresentation")):
            self.repString = "visualReferenceRepresentation"
        elif (node.isDefined("pinholeRepresentation")):
            self.repString = "pinholeRepresentation"
        elif (node.isDefined("sphericalRepresentation")):
            self.repString = "sphericalRepresentation"
        elif (node.isDefined("cylindricalRepresentation")):
            self.repString = "cylindricalRepresentation"
        self.repNode = node[self.repString]

        # type
        self.imageType = ""
        self.imageFileExt = ""
        if (self.repNode.isDefined("jpegImage")):
            self.imageFileExt = "jpg"
            self.imageType = "jpegImage"
        elif (self.repNode.isDefined("pngImage")):
            self.imageFileExt = "png"
            self.imageType = "pngImage"
        
        # bytes
        self.blob = self.repNode[self.imageType]
        self.bytes = self.blob.read_buffer()

        # pose
        self.translation = np.zeros(3)
        self.quaternion = np.zeros(4)
        if (self.node.isDefined("pose")):
            self.poseNode = self.node["pose"]
            self.translationNode = self.poseNode["translation"]
            self.rotationNode = self.poseNode["rotation"]
            self.translation[0] = self.translationNode["x"].value()
            self.translation[1] = self.translationNode["y"].value()
            self.translation[2] = self.translationNode["z"].value()
            self.quaternion[0] = self.rotationNode["x"].value()
            self.quaternion[1] = self.rotationNode["y"].value()
            self.quaternion[2] = self.rotationNode["z"].value()
            self.quaternion[3] = self.rotationNode["w"].value()
        
        # quaternion to matrix
        self.rotationMatrix = quaternion_to_matrix3x3(self.quaternion)

        # image properties
        for att in ["imageHeight", "imageWidth", "pixelHeight", "pixelWidth", "focalLength", "principalPointX", "principalPointY"]:
            if (self.repNode.isDefined(att)):
                print(f"{att}: {self.repNode[att].value()}")
        
        self.imageWidth = self.repNode["imageWidth"].value() if self.repNode.isDefined("imageWidth") else None
        self.imageHeight = self.repNode["imageHeight"].value() if self.repNode.isDefined("imageHeight") else None
        self.pixelWidth = self.repNode["pixelWidth"].value() if self.repNode.isDefined("pixelWidth") else None
        self.pixelHeight = self.repNode["pixelHeight"].value() if self.repNode.isDefined("pixelHeight") else None
        self.focalLength = self.repNode["focalLength"].value() if self.repNode.isDefined("focalLength") else None
        self.principalPointX = self.repNode["principalPointX"].value() if self.repNode.isDefined("principalPointX") else None
        self.principalPointY = self.repNode["principalPointY"].value() if self.repNode.isDefined("principalPointY") else None

        # sensor info                
        self.sensorVendor = self.node["sensorVendor"].value() if self.node.isDefined("sensorVendor") else None
        self.sensorModel = self.node["sensorModel"].value() if self.node.isDefined("sensorModel") else None

    def writeImageBytes(self, path: str):
        self.imageFilePath = f'{path}/{self.guid}.{self.imageFileExt}'
        with open(self.imageFilePath, 'wb') as imgFile:
            imgFile.write(self.bytes)

    def writeMetadata(self, path: str):
        self.metadataFilePath = f'{path}/{self.guid}.json'
        with open(self.metadataFilePath, 'w') as json_file:
            data = {
                "img" : self.guid,
                "px": self.translation[0],
                "py": self.translation[1],
                "pz": self.translation[2],
                "r00": self.rotationMatrix[0][0],
                "r01": self.rotationMatrix[0][1],
                "r02": self.rotationMatrix[0][2],
                "r10": self.rotationMatrix[1][0],
                "r11": self.rotationMatrix[1][1],
                "r12": self.rotationMatrix[1][2],
                "r20": self.rotationMatrix[2][0],
                "r21": self.rotationMatrix[2][1],
                "r22": self.rotationMatrix[2][2],
                "fx": self.imageWidth * self.focalLength if self.focalLength is not None else 0,
                "fy": self.imageHeight * self.focalLength if self.focalLength is not None else 0,
                "ox": self.principalPointX if self.principalPointX is not None else 0,
                "oy": self.principalPointY if self.principalPointY is not None else 0
            }

            json_data = json.dumps(data, indent=4)
            json_file.write(json_data)

def quaternion_to_matrix3x3(q: list[float]) -> np.array:
    # https://stackoverflow.com/questions/1556260/convert-quaternion-rotation-to-rotation-matrix
    qx = np.double(q[0])
    qy = np.double(q[1])
    qz = np.double(q[2])
    qw = np.double(q[3])

    n = 1.0 / np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx = qx * n
    qy = qy * n
    qz = qz * n
    qw = qw * n

    m = np.empty([3, 3])

    m[0][0] = 1.0 - 2.0 * qy * qy - 2.0 * qz * qz
    m[0][1] = 2.0 * qx * qy - 2.0 * qz * qw
    m[0][2] = 2.0 * qx * qz + 2.0 * qy * qw
    m[1][0] = 2.0 * qx * qy + 2.0 * qz * qw
    m[1][1] = 1.0 - 2.0 * qx * qx - 2.0 * qz * qz
    m[1][2] = 2.0 * qy * qz - 2.0 * qx * qw
    m[2][0] = 2.0 * qx * qz - 2.0 * qy * qw
    m[2][1] = 2.0 * qy * qz + 2.0 * qx * qw
    m[2][2] = 1.0 - 2.0 * qx * qx - 2.0 * qy * qy

    return m

def unpack(input_path: str, separate_scans: bool = True, front_views_only: bool = False):
    
    output_directory = os.path.splitext(input_path)[0] + r"-out/"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    e57 = pye57.E57(input_path)

    imf = e57.image_file
    root = imf.root()
    data2d = root["images2D"]
    count = data2d.childCount()

    if separate_scans:
        for i in range(0, count, 6):
            scanNmb = (i//6)+1
            print(f"\nScan: {scanNmb}/{count//6}")
            scanPath = output_directory + f'/scan{scanNmb}'

            if not os.path.exists(scanPath):
                os.makedirs(scanPath)

            nodes = {}
            if (front_views_only):
                if i + 1 < count:
                    nodes[i+1] = data2d[i+1]
            else:
                for j in range(0, 6):
                    if i + j < count:
                        nodes[i+j] = data2d[i+j]

            for k,v in nodes.items():
                print(f"\nExtracting node: {k+1}/{count}")
                img = MatterportImage(v)
                img.writeImageBytes(scanPath)
                img.writeMetadata(scanPath)
    else:
        for i in range(0, count):
            if (front_views_only and i%6 != 1):
                continue
            
            print(f"\nExtracting node: {i+1}/{count}")
            img = MatterportImage(data2d[i])
            img.writeImageBytes(output_directory)
            img.writeMetadata(output_directory)

if __name__ == '__main__':

    # 1. Install all dependencies mentioned at the top of the file with pip
    # 2. Acquire an e57 file of the desired Matterport scan
    # 3. Set input_path to point to the e57 file
    # 4. Run the script
    # 5. Check the "out" directory generated in the same directory as the e57

    input_path = r"/data/matterport/my_matterport_scan.e57"

    unpack(input_path)
