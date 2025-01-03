# Hexagon RealSLAM Scanner

## Step 1. Unpack .e57 files
open `https://github.com/immersal/immersal-python-tools-for-customer/tree/main/utils/unpack_e57.py`, make sure to enable this line:
```py
#For others, run this:
unpack(input_path, False, False)
```
run this script to unpack e57 data.

## Step 2. Submit the result directory which contains images and camera poses(json files).
run `submit_realslam20.py` or `submit_realslam10.py`
When it is completed, you should be able to find your map on the UI of Immersal developer portal.

**Note**
DO NOT run multiple submission scripts (Step 2) in parallel. Each execution clears the workspace for your account on the server. Running them concurrently can result in data overlap, leading to incorrect map generation.


