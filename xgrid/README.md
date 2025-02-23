# XGRID Scanner

Compatible with XGRID K1, L2 scanners.

## Step 1. Check the pose data
You need to make sure the head line of the pose data(csv file) is `# timestamp imgname TX TY TZ QX QY QZ QW`, e.g.
```csv
# timestamp imgname TX TY TZ QX QY QZ QW
1736408017.628399 pano/1736408017.628399.jpg 0.503564 -0.650374 0.836759 -0.657117 0.248333 -0.185543 0.687096
1736408022.625301 pano/1736408022.625301.jpg 1.167059 -0.804034 0.809150 -0.677060 0.228781 -0.162685 0.680281
1736408024.324244 pano/1736408024.324244.jpg 2.194855 -0.949470 0.774036 -0.689676 0.220652 -0.037252 0.688674
```
## Step 2. Submit the result directory which contains images and camera poses(csv file).
run `submit_xgrid.py`
When it is completed, you should be able to find your map on the UI of Immersal developer portal.

**Note**
DO NOT run multiple submission scripts (Step 2) in parallel. Each execution clears the workspace for your account on the server. Running them concurrently can result in data overlap, leading to incorrect map generation.


