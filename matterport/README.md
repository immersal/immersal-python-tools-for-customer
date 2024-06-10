

# Matterport support.

1. Since Immersal VPS requires multiple images + camera pose to construct a spatial map, you need to take more shot than usual. It is the recommended to take one shot every 0.5m, at least every 1m - 1.5m.

2. Check the 'main' function in the script. Specify url, token, map_name and input_directory.

3. You may specify advance parameters in data.
```py
data = {
    "token": token,
    "name": map_name,
    "featureCount": params.featureCount,
    "preservePoses": params.preservePoses,
}
```

Please refer to [Matterport Pipeline document](https://developers.immersal.com/docs/mapsmapping/advanced/metterport-pipeline/) for detail.