# NavVis Scanner

Check the data you get from NavVis, there are two cases:
1. If your data is a directory containing multiple images and a csv file of camera poses, use `submit_navvis_csv.py`.
2. If your data is a directory containing multiple images and each image has a corresponding json file of camera poses, use `submit_navvis_json.py`.

## Case 1.
You need to make sure the head line of the pose data(csv file) is `ID; filename; timestamp; pano_pos_x; pano_pos_y; pano_pos_z; pano_ori_w; pano_ori_x; pano_ori_y; pano_ori_z`, e.g.
```csv
ID; filename; timestamp; pano_pos_x; pano_pos_y; pano_pos_z; pano_ori_w; pano_ori_x; pano_ori_y; pano_ori_z
0; 00000-pano.jpg; 1647329314.715836; -0.599143; -0.587440; 1.886546; 0.746558; -0.005474; -0.028297; 0.664696
1; 00001-pano.jpg; 1647329318.949630; 1.777497; 0.314116; 1.878300; 0.754395; -0.022532; -0.013212; 0.655901
2; 00002-pano.jpg; 1647329327.291312; 1.769323; 3.316548; 2.865397; 0.695152; 0.002385; 0.029978; 0.718234
```
Then specify the image directory and the pose data file, run `submit_navvis_csv.py`.

## Case 2.
Specify the directory containing multiple images and each image has a corresponding json file of camera poses, run `submit_navvis_json.py`.

**Note**
DO NOT run multiple submission scripts in parallel. Each execution clears the workspace for your account on the server. Running them concurrently can result in data overlap, leading to incorrect map generation.


