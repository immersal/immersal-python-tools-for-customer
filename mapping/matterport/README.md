

# Matterport Support.

**Note** Since Immersal VPS requires multiple images + camera pose to construct a spatial map, you need to take more shot than usual. It is the recommended to take one shot every 0.5m, at least every 1m - 1.5m. Please refer to [Matterport Pipeline document](https://developers.immersal.com/docs/mapsmapping/advanced/metterport-pipeline/) for detail.

## Step 1. Unpack .e57 files
open `https://github.com/immersal/immersal-python-tools-for-customer/tree/main/utils/unpack_e57.py`, make sure to enable this line:
```py
#For matterport, run this:
unpack(input_path)
```

## Step 2. Submitting data
Check the 'main' function in the script. Specify url, token, map_name and input_directory.
run script `submit_matterport.py`

