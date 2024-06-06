import os
import json
import numpy as np      # pip intall numpy
import cv2              # pip install opencv-python



class Intrinsics():
    
    def __init__(self, fx: float, fy: float, ox: float, oy: float):
        self.fx = fx
        self.fy = fy
        self.ox = ox
        self.oy = oy


def main(input_videos: list[str],  intrinsics: Intrinsics=None, output_directory: str='out', prefix: str="", interval_seconds: float=1.0, cv_rotate: int=-1) -> None:

    curr_frame = 0

    for video in input_videos:

        print(video)

        if not os.path.exists(video):
            print(f"video {video} not found")
            return

        if not os.path.exists(output_directory):
            print(f'created output directory {output_directory}')
            os.makedirs(output_directory)

        vid = cv2.VideoCapture(video)
        framerate = vid.get(cv2.CAP_PROP_FPS)
        total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

        # list of frames to extract, no need to play through the whole video
        frames = np.arange(0, total_frames, framerate * interval_seconds)
        milliseconds = list(map(lambda x: x / framerate * 1000, frames))

        for i, ms in enumerate(milliseconds):
            vid.set(cv2.CAP_PROP_POS_MSEC, ms)
            retval, img = vid.retrieve()
            height, width, channels = img.shape

            if retval:
                if cv_rotate > -1:
                    # ROTATE_90_CLOCKWISE        = 0,
                    # ROTATE_180                 = 1,
                    # ROTATE_90_COUNTERCLOCKWISE = 2,
                    img = cv2.rotate(img, cv_rotate)

                output_filepath = os.path.join(output_directory, f'{prefix}_frame_{curr_frame:04d}.png')
                print(f'saving {output_filepath}')
                cv2.imwrite(output_filepath, img)

                # cv2.imshow('img', img)
                # cv2.waitKey(500)
                # cv2.destroyAllWindows()

                if(intrinsics != None):
                    json_path = output_filepath.replace('.png', '.json')
                    with open(json_path, 'w') as json_file:
                        data = {
                            "img": os.path.basename(output_filepath),
                            "fx": intrinsics.fx,
                            "fy": intrinsics.fy,
                            "ox": intrinsics.ox,
                            "oy": intrinsics.oy,
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
                            # "r00": 0.0, 
                            # "r01": 1.0, 
                            # "r02": 0.0, 
                            # "r10": 1.0, 
                            # "r11": 0.0, 
                            # "r12": 0.0, 
                            # "r20": 0.0,
                            # "r21": 0.0, 
                            # "r22": -1.0,
                        }

                        json_data = json.dumps(data, indent=4)
                        json_file.write(json_data)

                curr_frame += 1
        vid.release()


if __name__ == '__main__':
    
    # Path of your videos, you may pass multiple ones, e.g.
    # input_videos = ["/Users/maolin/workspaces/mapping-360/video1.mp4",
    #                 "/Users/maolin/workspaces/mapping-360/video2.mp4",
    #                 "/Users/maolin/workspaces/mapping-360/video3.mp4"]
    input_videos = ["path_of_video"] 
    
    # Path of the directory of output frames
    # e.g. output_directory = "/Users/maolin/workspaces/mapping-360/GS010023/frames"
    output_directory = "path_of_output_frames"
    
    # make sure you know the intrinsics. For 360 camera, fill (0,0,0,0).
    # intrinsics = Intrinsics(1640.078, 1640.078, 967.280, 536.117) #iPhone 13 Pro 1x video with 60fps HD
    intrinsics = Intrinsics(0, 0, 0, 0)

    main(input_videos, intrinsics=intrinsics, output_directory=output_directory, prefix="loc1_", interval_seconds=1, cv_rotate=-1)

    # NO_ROTATION                =-1,
    # ROTATE_90_CLOCKWISE        = 0,
    # ROTATE_180                 = 1,
    # ROTATE_90_COUNTERCLOCKWISE = 2,




