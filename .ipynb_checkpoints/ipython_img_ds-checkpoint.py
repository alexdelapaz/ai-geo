 1/1: /datasets/PXL_20240418_181519812.MP.jpg
 1/2: ls
 1/3: import os
 1/4: os.listdir()
 1/5: pwd
 1/6: cd /datasets/
 1/7: import exifread
 1/8:
exd = exifread.process_file(from PIL import Image
,details=False)
 1/9: from PIL import Image
1/10: Image.open('PXL_20240418_181519812.MP')
1/11: Image.open('PXL_20240418_181519812.MP.jpg')
1/12: img = Image.open('PXL_20240418_181519812.MP.jpg')
1/13: img
1/14: print(img)
1/15: exd = exifread.process_file(img)
1/16: exif_data = img._getexif()
1/17: exif_data
1/18: from PIL.ExifTags import TAGS
1/19:
for tag_i, value in exif_data.items():
    tag_name = TAGS.get(tag_i, , tag_i)
1/20:
for tag_i, value in exif_data.items():
    tag_name = TAGS.get(tag_i, tag_i)
    print(f'{tag_name}: {value}')
1/21: TAGS
1/22:
for tag_i, value in exif_data.items():
    tag_name = TAGS.get(tag_i, tag_i)
    print(f'{tag_name}: {value}')
1/23: exif
1/24: exif_data
1/25:
for tag_i, value in exif_data.items():
    tag_name = TAGS.get(tag_i, tag_i)
    print(f'{tag_name}: {value}')
1/26: import exifread
1/27: exifread.process_file(img, details=False)
1/28:
with open('./PXL_20240418_181519812.MP') as img_file:
        tags = exifread.process_file(img_file, details=False)
1/29:
with open('./PXL_20240418_181519812.MP.jpg') as img_file:
        tags = exifread.process_file(img_file, details=False)
1/30:
with open('./PXL_20240418_181519812.MP.jpg', 'rb') as img_file:
        tags = exifread.process_file(img_file, details=False)
1/31: tags
1/32: tags['GPS GPSLatitudeRef']
1/33: tags['GPS GPSLatitudeRef'].values
1/34: tags['GPS GPSLatitude'].values
1/35: tags['GPS GPSLatitude'].values.printable
1/36: tags.items()
1/37: type(tags)
1/38: tags['GPS GPSLatitude'].printable
1/39: tags['GPS GPSLatitude'].values
1/40: tags['GPS GPSLatitude'].values[0]
1/41: tags['GPS GPSLatitude'].printable[0]
1/42: tags
1/43:
with open('./PXL_20240418_181519812.MP.jpg', 'rb') as img_file:
        tags = exifread.process_file(img_file, details=False)
1/44:
for tag_i, value in exif_data.items():
    tag_name = TAGS.get(tag_i, tag_i)
    print(f'{tag_name}: {value}')
1/45: ls
1/46: ls /
1/47: %history -f './datasets/ipython_geo_ds.py'
1/48: %history -f ./datasets/ipython_geo_ds.py
1/49: %history -f /datasets/ipython_geo_ds.py
1/50: cat /datasets/ipython_geo_ds.py
1/51:
for tag_i, value in exif_data.items():
    tag_name = TAGS.get(tag_i, tag_i)
    print(f'{tag_name}: {value}')
1/52: exif_data
1/53:
for tag_i, value in exif_data.items():
    tag_name = TAGS.get(tag_i, tag_i)
    print(f'{tag_name}: {value}')
1/54: exif_data[TAGS.get('GPSInfo')]
1/55: TAGS.get('GPSInfo')
1/56: TAGS
1/57:
for tag_i, value in exif_data.items():
    tag_name = TAGS.get(tag_i, tag_i)
    print(f'{tag_name}: {value}')
1/58:
for tag_i, value in exif_data.items():
    tag_name = TAGS.get(tag_i, tag_i)
    print(f'{tag_name},{tag_i}: {value}')
1/59: TAGS.get(34853)
1/60: exif_data[TAGS.get(34853)]
1/61: exif_data[34853]
1/62:
version_bytes = exif_data[34853][0]
version_numbers = tuple(version_bytes)  # (2, 2, 0, 0)
print(f"GPS EXIF Version: {version_numbers[0]}.{version_numbers[1]}")
1/63: exif_data[34853][0]
1/64: tuple(exif_data[34853][0])
1/65: exif_data[34853][0]
1/66: exif_data[34853][0][0]
1/67: exif_data[34853][0][1]
1/68: type(exif_data[34853][0])
1/69: print(b'\x68\x65\x6C\x6C\x6F')
1/70: print('\x68\x65\x6C\x6C\x6F')
1/71: print('\x68')
1/72: b'/x68'
1/73: b'\x68'
1/74: b'\x02'
1/75: b'\x02'
1/76: print(b'\x02')
1/77: b'\x02'
1/78: exif_data[34853][0][0]
1/79: exif_data[34853][0]
1/80: b'\x02\x02\x00\x00'
1/81: b'\x02'
1/82: print(b'\x02')
1/83: print(tuple(b'\x02'))
1/84: print(tuple(b'\x41'))
1/85: clear
1/86: cat /datasets/ipython_image_ds.py
1/87: %history -f /datasets/ipython_img_ds.py
 2/1: %history -f /datasets/ipython_img_ds.py
 2/2: ls
 2/3: ls /datasets/
 2/4: rm /datasets/ipython_img_ds.py
 2/5: %history -f ./ipython_img_ds.py
 2/6: pwd
 2/7: ls
 2/8: pwd
 4/1: import requests
 4/2:
cog_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/33/U/UP/2024/4/21/S2A_33UUP_20240421_0_L2A/B04.tif"
local_filename = "sentinel_band04.tif"

response = requests.get(cog_url, stream=True)
response.raise_for_status()

with open(local_filename, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"Downloaded COG file to {local_filename}")
 4/3:
import requests

# Define search parameters
search_url = "https://earth-search.aws.element84.com/v0/search"
params = {
    "collections": ["sentinel-s2-l2a-cogs"],
    "datetime": "2024-04-21T00:00:00Z/2024-04-21T23:59:59Z",
    "bbox": [12.0, 41.0, 13.0, 42.0],  # Replace with your area of interest
    "limit": 1
}

# Perform the search
response = requests.post(search_url, json=params)
search_results = response.json()

# Extract asset URLs
for feature in search_results["features"]:
    assets = feature["assets"]
    for band, asset_info in assets.items():
        print(f"{band}: {asset_info['href']}")
 4/4: search_results
 4/5:
import requests

# Define search parameters
search_url = "https://earth-search.aws.element84.com/v0/search"
params = {
    "collections": ["sentinel-s2-l2a-cogs"],
    "datetime": "2024-04-21T00:00:00Z/2024-04-21T23:59:59Z",
    "bbox": [36,72,36.5,72.5],  # Replace with your area of interest
    "limit": 1
}

# Perform the search
response = requests.post(search_url, json=params)
search_results = response.json()

# Extract asset URLs
for feature in search_results["features"]:
    assets = feature["assets"]
    for band, asset_info in assets.items():
        print(f"{band}: {asset_info['href']}")
 4/6: search_results
 4/7:
import requests

# Define search parameters





search_url = "https://earth-search.aws.element84.com/v0/search"
params = {
    "collections": ["sentinel-s2-l2a-cogs"],
    "datetime": "2024-04-21T00:00:00Z/2024-04-21T23:59:59Z",
    "bbox": [36,72,36.5,72.5],  # Replace with your area of interest
    "limit": 1
}

# Perform the search
response = requests.post(search_url, json=params)
search_results = response.json()

# Extract asset URLs
for feature in search_results["features"]:
    assets = feature["assets"]
    for band, asset_info in assets.items():
        print(f"{band}: {asset_info['href']}")
 4/8:
%matplotlib inline
import intake
import satsearch
 4/9: pip install matplotlib
4/10:
%matplotlib inline
import intake
import satsearch
4/11:
import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Specify the path for Landsat TIF on AWS
fp = 'http://landsat-pds.s3.amazonaws.com/c1/L8/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'

# See the profile
with rasterio.open(fp) as src:
    print(src.profile)
4/12: pip install rasterio
4/13:
import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Specify the path for Landsat TIF on AWS
fp = 'http://landsat-pds.s3.amazonaws.com/c1/L8/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'

# See the profile
with rasterio.open(fp) as src:
    print(src.profile)
4/14:
import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Specify the path for Landsat TIF on AWS
fp = 'http://landsat-pds.s3.amazonaws.com/c1/L8/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'

# See the profile
with rasterio.open(fp) as src:
    print(src.profile)print('Landsat on Google:')
filepath = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
with rasterio.open(filepath) as src:
    print(src.profile)
4/15:
print('Landsat on Google:')
filepath = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
with rasterio.open(filepath) as src:
    print(src.profile)
4/16:
# The grid of raster values can be accessed as a numpy array and plotted:
with rasterio.open(filepath) as src:
   oviews = src.overviews(1) # list of overviews from biggest to smallest
   oview = oviews[-1] # let's look at the smallest thumbnail
   print('Decimation factor= {}'.format(oview))
   # NOTE this is using a 'decimated read' (http://rasterio.readthedocs.io/en/latest/topics/resampling.html)
   thumbnail = src.read(1, out_shape=(1, int(src.height // oview), int(src.width // oview)))

print('array type: ',type(thumbnail))
print(thumbnail)

plt.imshow(thumbnail)
plt.colorbar()
plt.title('Overview - Band 4 {}'.format(thumbnail.shape))
plt.xlabel('Column #')
plt.ylabel('Row #')
4/17:
# The grid of raster values can be accessed as a numpy array and plotted:
with rasterio.open(filepath) as src:
   oviews = src.overviews(1) # list of overviews from biggest to smallest
   oview = oviews[-1] # let's look at the smallest thumbnail
   print('Decimation factor= {}'.format(oview))
   # NOTE this is using a 'decimated read' (http://rasterio.readthedocs.io/en/latest/topics/resampling.html)
   thumbnail = src.read(1, out_shape=(1, int(src.height // oview), int(src.width // oview)))

print('array type: ',type(thumbnail))
print(thumbnail)

plt.imshow(thumbnail)
plt.colorbar()
plt.title('Overview - Band 4 {}'.format(thumbnail.shape))
plt.xlabel('Column #')
plt.ylabel('Row #')
4/18:
print('Landsat on Google:')
filepath = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
with rasterio.open(filepath) as src:
    print(src.oviews)
4/19:
print('Landsat on Google:')
filepath = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
with rasterio.open(filepath) as src:
    print(src.overviews)
4/20:
# The grid of raster values can be accessed as a numpy array and plotted:
with rasterio.open(filepath) as src:
   oviews = src.overviews(1) # list of overviews from biggest to smallest
   oview = oviews[-1] # let's look at the smallest thumbnail
4/21:
# The grid of raster values can be accessed as a numpy array and plotted:
with rasterio.open(filepath) as src:
   oviews = src.overviews(1) # list of overviews from biggest to smallest
   oview = oviews[-1] # let's look at the smallest thumbnail
4/22:
# The grid of raster values can be accessed as a numpy array and plotted:
with rasterio.open(filepath) as src:
   oviews = src.overviews(1) # list of overviews from biggest to smallest
   oview = oviews
4/23: oview
4/24:
# The grid of raster values can be accessed as a numpy array and plotted:
with rasterio.open(filepath) as src:
   oviews = src.overviews(1) # list of overviews from biggest to smallest
   print(oview)
4/25:
# The grid of raster values can be accessed as a numpy array and plotted:
with rasterio.open(filepath) as src:
   oviews = src.overviews(1) # list of overviews from biggest to smallest
   print(oviews)
4/26:
# The grid of raster values can be accessed as a numpy array and plotted:
with rasterio.open(filepath) as src:
   oviews = src.overviews(1) # list of overviews from biggest to smallest
   print(oviewsss)
4/27:
# The grid of raster values can be accessed as a numpy array and plotted:
with rasterio.open(filepath) as src:
   oviews = src.overviews(1) # list of overviews from biggest to smallest
   print(len(oviews))
4/28:
# The grid of raster values can be accessed as a numpy array and plotted:
with rasterio.open(filepath) as src:
   oviews = src.overviews(0) # list of overviews from biggest to smallest
   print(len(oviews))
4/29:
# The grid of raster values can be accessed as a numpy array and plotted:
with rasterio.open(filepath) as src:
   print(src.overviews(1))
4/30:
print('Landsat on Google:')
filepath = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
with rasterio.open(filepath) as src:
    print(src.overviews)
4/31:
print('Landsat on Google:')
filepath = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
with rasterio.open(filepath) as src:
    print(src.overviews(!))
4/32:
print('Landsat on Google:')
filepath = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
with rasterio.open(filepath) as src:
    print(src.overviews(1))
4/33: filepath = 'http://landsat-pds.s3.amazonaws.com/c1/L8/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
4/34:
print('Landsat on Google:')
filepath = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
with rasterio.open(filepath) as src:
    print(src.overviews(1))
4/35:
filepath = 'http://landsat-pds.s3.amazonaws.com/c1/L8/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
with rasterio.open(filepath) as src:
    print(src.overviews(1))
4/36: filepath = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
4/37:
with rasterio.open(filepath) as src:
    print(src.overviews(1))
4/38:
with rasterio.open(filepath) as src:
    print(src.overviews)
4/39:
with rasterio.open(filepath) as src:
    print(src.overviews(0))
4/40:
filepath = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
4/41:
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
4/42: cog_url  = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
4/43:
with rasterio.open(cog_url) as dataset:
    print(dataset.profile)

    # Read the raster data
    raster_data = dataset.read(1)  # Reading the first band

    # Display the raster data
    plt.figure(figsize=(10, 10))
    show(raster_data, cmap='gray', title='Sentinel-2 Band 4')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()
4/44: plt.show()
4/45: pwd
4/46: ls /
4/47:
plt.figure(figsize=(10, 10))
show(raster_data, cmap='gray')
plt.savefig('/datasetssentinel_band4.png', dpi=300, bbox_inches='tight')
4/48: ls
4/49: pwd
4/50: ls /
4/51: rm /datasetssentinel_band4.png
4/52: ls /
4/53:
plt.figure(figsize=(10, 10))
show(raster_data, cmap='gray')
plt.savefig('/datasets/sentinel_band4.png', dpi=300, bbox_inches='tight')
4/54:
with rasterio.open(cog_url) as dataset:
    print(dataset.profile)

    # Read the raster data
    raster_data = dataset.read(1)  # Reading the first band

    # Display the raster data
    plt.figure(figsize=(10, 10))
    show(raster_data, cmap='gray', title='Sentinel-2 Band 4')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()
4/55:
plt.figure(figsize=(10, 10))
show(raster_data, cmap='gray')
plt.savefig('/datasets/sentinel_band4.png', dpi=300, bbox_inches='tight')
4/56:

# Open COG using rasterio
with rasterio.open(cog_url) as dataset:
    print(dataset.profile)

    # Read the raster data
    band = dataset.read(1).astype('float32')

    # Handle no-data values
    band[band == dataset.nodata] = np.nan

    # Normalize the data (common Sentinel-2 scaling factor: 10000)
    band /= 10000.0

    # Clip values to enhance visual contrast
    vmin, vmax = np.nanpercentile(band, (2, 98))

    plt.figure(figsize=(10, 10))
    plt.imshow(band, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Sentinel-2 Band 4 (scaled)')
    plt.colorbar(label='Reflectance')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig('sentinel_band4.png', dpi=300, bbox_inches='tight')
    plt.show()
4/57:
plt.figure(figsize=(10, 10))
show(raster_data, cmap='gray')
plt.savefig('/datasets/sentinel_band4.png', dpi=300, bbox_inches='tight')
4/58:

# Open COG using rasterio
with rasterio.open(cog_url) as dataset:
    print(dataset.profile)

    # Read the raster data
    band = dataset.read(1).astype('float32')

    # Handle no-data values
    band[band == dataset.nodata] = np.nan

    # Normalize the data (common Sentinel-2 scaling factor: 10000)
    band /= 10000.0

    # Clip values to enhance visual contrast
    vmin, vmax = np.nanpercentile(band, (2, 98))

    plt.figure(figsize=(10, 10))
    plt.imshow(band, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Sentinel-2 Band 4 (scaled)')
    plt.colorbar(label='Reflectance')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig('/datasets/sentinel_band4_new.png', dpi=300, bbox_inches='tight')
4/59: cog_url  = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B#.TIF'
4/60: cog_url  = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B3.TIF'
4/61:

# Open COG using rasterio
with rasterio.open(cog_url) as dataset:
    print(dataset.profile)

    # Read the raster data
    band = dataset.read(1).astype('float32')

    # Handle no-data values
    band[band == dataset.nodata] = np.nan

    # Normalize the data (common Sentinel-2 scaling factor: 10000)
    band /= 10000.0

    # Clip values to enhance visual contrast
    vmin, vmax = np.nanpercentile(band, (2, 98))

    plt.figure(figsize=(10, 10))
    plt.imshow(band, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Sentinel-2 Band 4 (scaled)')
    plt.colorbar(label='Reflectance')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig('/datasets/sentinel_band3_new.png', dpi=300, bbox_inches='tight')
4/62:

# Open COG using rasterio
with rasterio.open(cog_url) as dataset:
    print(dataset.profile)

    # Read the raster data
    band = dataset.read(1).astype('float32')

    # Handle no-data values
    band[band == dataset.nodata] = np.nan

    # Normalize the data (common Sentinel-2 scaling factor: 10000)
    band /= 10000.0

    # Clip values to enhance visual contrast
    vmin, vmax = np.nanpercentile(band, (2, 98))

    plt.figure(figsize=(10, 10))
    plt.imshow(band, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Sentinel-2 Band 4 (scaled)')
    plt.colorbar(label='Reflectance')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig('/datasets/sentinel_band3_new.png', dpi=300, bbox_inches='tight')
4/63:

# Open COG using rasterio
with rasterio.open(cog_url) as dataset:
    print(dataset.profile)

    # Read the raster data
    band = dataset.read(1).astype('float32')

    # Handle no-data values
    band[band == dataset.nodata] = np.nan

    # Normalize the data (common Sentinel-2 scaling factor: 10000)
    band /= 10000.0

    # Clip values to enhance visual contrast
    vmin, vmax = np.nanpercentile(band, (2, 98))

    plt.figure(figsize=(10, 10))
    plt.imshow(band, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Sentinel-2 Band 4 (scaled)')
    plt.colorbar(label='Reflectance')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig('/datasets/sentinel_band3_new.png', dpi=300, bbox_inches='tight')
4/64: cog_url  = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B3.TIF'
4/65:

# Open COG using rasterio
with rasterio.open(cog_url) as dataset:
    print(dataset.profile)

    # Read the raster data
    band = dataset.read(1).astype('float32')

    # Handle no-data values
    band[band == dataset.nodata] = np.nan

    # Normalize the data (common Sentinel-2 scaling factor: 10000)
    band /= 10000.0

    # Clip values to enhance visual contrast
    vmin, vmax = np.nanpercentile(band, (2, 98))

    plt.figure(figsize=(10, 10))
    plt.imshow(band, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Sentinel-2 Band 4 (scaled)')
    plt.colorbar(label='Reflectance')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig('/datasets/sentinel_band3_new.png', dpi=300, bbox_inches='tight')
4/66:

# Open COG using rasterio
with rasterio.open(cog_url) as dataset:
    print(dataset.profile)

    # Read the raster data
    band = dataset.read(1).astype('float32')

    print(band)
4/67: filepath = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
4/68:

# Open COG using rasterio
with rasterio.open(cog_url) as dataset:
    print(dataset.profile)

    # Read the raster data
    band = dataset.read(1).astype('float32')

    print(band)
4/69: cog_url = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
4/70:

# Open COG using rasterio
with rasterio.open(cog_url) as dataset:
    print(dataset.profile)

    # Read the raster data
    band = dataset.read(1).astype('float32')

    print(band)
4/71: clear
4/72: cog_url = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
4/73:

# Open COG using rasterio
with rasterio.open(cog_url) as dataset:
    print(dataset.profile)

    try:
        for i in range(4):
            print(dataset.read(i))
    except:
        pass
    # Read the raster data
    band = dataset.read(1).astype('float32')

    print(band)
4/74:

# Open COG using rasterio
with rasterio.open(cog_url) as dataset:clear

    print(dataset.profile)

    # Read the raster data
    band = dataset.read(1).astype('float32')

    # Handle no-data values
    band[band == dataset.nodata] = np.nan

    # Normalize the data (common Sentinel-2 scaling factor: 10000)
    band /= 10000.0

    # Clip values to enhance visual contrast
    vmin, vmax = np.nanpercentile(band, (2, 98))

    plt.figure(figsize=(10, 10))
    plt.imshow(band, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Sentinel-2 Band 4 (scaled)')
    plt.colorbar(label='Reflectance')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig('/datasets/sentinel_band3_new.png', dpi=300, bbox_inches='tight')
4/75: clear
4/76: src = rio.open(cog_url)
4/77: import rasterio as rio
4/78: src = rio.open(cog_url)
4/79: src
4/80: src.crs
4/81: src.profile
4/82: src.crs
4/83: type(src.crs)
4/84: type(src.profile)
4/85: type(src.profile.keys())
4/86: type(src.profile.keys()[0])
4/87: type(list(src.profile.keys()))
4/88: type(list(src.profile.keys())[0])
4/89: list(src.profile.keys())[0]
4/90: src.profile
4/91:
for feature in search_results["features"]:
    print(feature)
4/92:
import requests

# Define search parameters





search_url = "https://earth-search.aws.element84.com/v0/search"
params = {
    "collections": ["sentinel-s2-l2a-cogs"],
    "datetime": "2024-04-21T00:00:00Z/2024-04-21T23:59:59Z",
    "bbox": [36,72,36.5,72.5],  # Replace with your area of interest
    "limit": 1
}

# Perform the search
response = requests.post(search_url, json=params)
search_results = response.json()

# Extract asset URLs
for feature in search_results["features"]:
    assets = feature["assets"]
    for band, asset_info in assets.items():
        print(f"{band}: {asset_info['href']}")
4/93: src.profil
4/94: src.prf
4/95: src.profile
4/96: src.crs
4/97: src.read(0)
4/98: src.read(1)
4/99: src.read(2)
4/100: src.read(3)
4/101: src.read(1)
4/102: src.profil
4/103: src.profile
4/104: in = src.read(1)
4/105: corg = src.read(1)
4/106: cog = src.read(1)
4/107: src = rio.open(cog_url)
4/108: cog_url
4/109: cog = src.read(1)
4/110: cog_url
4/111: src = rio.open(cog_url)
4/112: cog = src.read(1)
4/113:
import rasterio as rio
from rasterio.session import AWSSession
import boto3
from rasterio.env import Env

session = boto3.Session()
aws_session = AWSSession(session)

config = {
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
    'GDAL_HTTP_MAX_RETRY': '5',
    'GDAL_HTTP_RETRY_DELAY': '2'
}

with Env(session=aws_session, **config):
    with rio.open(cog_url) as src:
        cog = src.read(1)
4/114:
import rasterio as rio
from rasterio.env import Env

cog_url = '/vsicurl/https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'

config = {
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
    'GDAL_HTTP_MAX_RETRY': '5',
    'GDAL_HTTP_RETRY_DELAY': '2'
}

with Env(**config):
    with rio.open(cog_url) as src:
        cog = src.read(1)
4/115:
filepath = 'http://landsat-pds.s3.amazonaws.com/c1/L8/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
with rasterio.open(filepath) as src:
    print(src.overviews(1))
4/116:
print('Landsat on Google:')
filepath = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'
with rasterio.open(filepath) as src:
    print(src.overviews(1))
4/117: cog_url_1 = 's3://usgs-landsat/collection02/level-2/standard/oli-tirs/2022/016/041/LC09_L2SP_016041_20221022_20230325_02_T1/LC09_L2SP_016041_20221022_20230325_02_T1_SR_B2.TIF'
4/118:
import rasterio as rio
from rasterio.env import Env

cog_url = '/vsicurl/https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'

config = {
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
    'GDAL_HTTP_MAX_RETRY': '5',
    'GDAL_HTTP_RETRY_DELAY': '2'
}

with Env(**config):
    with rio.open(cog_url) as src:
        cog = src.read(1)
4/119: src = rio.open(cog_url_1)
4/120:
import rasterio as rio
from rasterio.env import Env

cog_url_1 = 'https://landsat-pds.s3.amazonaws.com/c1/L8/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'

# use /vsicurl/ explicitly
cog_url_1 = f'/vsicurl/{cog_url_1}'

with Env(AWS_NO_SIGN_REQUEST='YES'):
    with rio.open(cog_url_1) as src:
        cog = src.read(1)

print(cog.shape)
4/121:
import rasterio as rio
from rasterio.env import Env

cog_url_1 = 'https://landsat-pds.s3.amazonaws.com/c1/L8/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'

# use /vsicurl/ explicitly
cog_url_1 = f'/vsicurl/{cog_url_1}'

with Env(AWS_NO_SIGN_REQUEST='YES'):
    with rio.open(cog_url_1) as src:
        cog = src.read(1)

print(cog.shape)
4/122:
curl /vsicurl/https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1
        ⋮ /LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF
4/123:
import rasterio as rio
from rasterio.env import Env

url = '/vsicurl/https://landsat-pds.s3.amazonaws.com/c1/L8/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'

with Env(AWS_NO_SIGN_REQUEST='YES'):
    with rio.open(url) as src:
        cog = src.read(1)

print(cog.shape)
4/124: pip install boto3
4/125:
import rasterio as rio
from rasterio.session import AWSSession
import boto3
from rasterio.env import Env

session = boto3.Session()
aws_session = AWSSession(session)

config = {
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
    'GDAL_HTTP_MAX_RETRY': '5',
    'GDAL_HTTP_RETRY_DELAY': '2'
}

with Env(session=aws_session, **config):
    with rio.open(cog_url) as src:
        cog = src.read(1)
4/126:
import rasterio as rio
from rasterio.env import Env

cog_url = '/vsicurl/https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'

config = {
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.TIF',
    'GDAL_HTTP_MAX_RETRY': '10',
    'GDAL_HTTP_RETRY_DELAY': '5',
    'GDAL_HTTP_TIMEOUT': '30'
}

with Env(**config):
    with rio.open(cog_url) as src:
        cog = src.read(1, masked=True)  # `masked=True` can handle partial reads better

print(cog.shape)
4/127:
import rasterio as rio
from rasterio.env import Env

cog_url = '/vsicurl/https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'

config = {
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.TIF',
    'GDAL_HTTP_MAX_RETRY': '10',
    'GDAL_HTTP_RETRY_DELAY': '5',
    'GDAL_HTTP_TIMEOUT': '30'
}

with Env(**config):
    with rio.open(cog_url) as src:
        cog = src.read(1, masked=True)  # `masked=True` can handle partial reads better

print(cog.shape)
4/128:
import requests
import rasterio as rio

url = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'

local_file = 'LC08_B4.tif'

# Robustly download entire file
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(local_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# Then open locally
with rio.open(local_file) as src:
    cog = src.read(1)

print(cog.shape)
4/129: cog
4/130:
    print("Filename:", src.name)
    print("Driver:", src.driver)
    print("Width, Height:", src.width, src.height)
    print("CRS:", src.crs)
    print("Transform (georeferencing):", src.transform)
    print("Number of Bands:", src.count)
    print("Data type:", src.dtypes)
    print("Bounds:", src.bounds)
    print("Compression:", src.profile.get("compress"))
4/131: src = cog
4/132:
    print("Filename:", src.name)
    print("Driver:", src.driver)
    print("Width, Height:", src.width, src.height)
    print("CRS:", src.crs)
    print("Transform (georeferencing):", src.transform)
    print("Number of Bands:", src.count)
    print("Data type:", src.dtypes)
    print("Bounds:", src.bounds)
    print("Compression:", src.profile.get("compress"))
4/133: cog.name
4/134: cog.driver
4/135:
import requests
import rasterio as rio

url = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'

local_file = 'LC08_B4.tif'

# Robustly download entire file
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(local_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# Then open locally
with rio.open(local_file) as src:
    cog = src.read(1)
    print("Filename:", src.name)
    print("Driver:", src.driver)
    print("Width, Height:", src.width, src.height)
    print("CRS:", src.crs)
    print("Transform (georeferencing):", src.transform)
    print("Number of Bands:", src.count)
    print("Data type:", src.dtypes)
    print("Bounds:", src.bounds)
    print("Compression:", src.profile.get("compress"))
print(cog.shape)
4/136: type(cog)
4/137: numpy
4/138: numpy.ndarray
4/139: np.ndarray
4/140: np.ndarray == type(cog)
4/141: print(cog.dtype)
4/142: print(cog)
4/143: print(cog.dims)
4/144: print(cog.dim)
4/145:
from PIL import Image
import numpy as np

# Normalize the data for 8-bit PNG (optional but usually necessary)
array = cog
array = array.astype(np.float32)
array = (255 * (array - array.min()) / (array.ptp())).astype(np.uint8)

# Save as PNG
Image.fromarray(array).save("/datasets/SAT_landsat_b4.png")
4/146: print(cog)
4/147: clear
4/148: print(cog)
4/149: print(cog.crs)
4/150: print(src.crs)
4/151: clear
4/152: print(src.crs)
4/153: src.crs
4/154: print(src.profile)
4/155:
import requests

# Define search parameters





search_url = "https://earth-search.aws.element84.com/v0/search"
params = {
    "collections": ["sentinel-s2-l2a-cogs"],
    "datetime": "2024-04-21T00:00:00Z/2024-04-21T23:59:59Z",
    "bbox": [36,72,36.5,72.5],  # Replace with your area of interest
    "limit": 1
}

# Perform the search
response = requests.post(search_url, json=params)
search_results = response.json()

# Extract asset URLs
for feature in search_results["features"]:
    assets = feature["assets"]
    for band, asset_info in assets.items():
        print(f"{band}: {asset_info['href']}")
4/156: response
4/157: response.text
4/158:

# Open COG using rasterio
with rasterio.open(cog_url) as dataset:clear

    print(dataset.profile)

    # Read the raster data
    band = dataset.read(1).astype('float32')

    # Handle no-data values
    band[band == dataset.nodata] = np.nan

    # Normalize the data (common Sentinel-2 scaling factor: 10000)
    band /= 10000.0

    # Clip values to enhance visual contrast
    vmin, vmax = np.nanpercentile(band, (2, 98))

    plt.figure(figsize=(10, 10))
    plt.imshow(band, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Sentinel-2 Band 4 (scaled)')
    plt.colorbar(label='Reflectance')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig('/datasets/SAT_sentinel_band_4.png', dpi=300, bbox_inches='tight')
4/159:

# Open COG using rasterio
with rasterio.open(cog_url) as dataset:clear

    print(dataset.profile)

    # Read the raster data
    band = dataset.read(1).astype('float32')

    # Handle no-data values
    band[band == dataset.nodata] = np.nan

    # Normalize the data (common Sentinel-2 scaling factor: 10000)
    band /= 10000.0

    # Clip values to enhance visual contrast
    vmin, vmax = np.nanpercentile(band, (2, 98))

    plt.figure(figsize=(10, 10))
    plt.imshow(band, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Sentinel-2 Band 4 (scaled)')
    plt.colorbar(label='Reflectance')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig('/datasets/SAT_sentinel_band_4.png', dpi=300, bbox_inches='tight')
4/160:

# Open COG using rasterio
with rasterio.open(cog_url) as dataset:clear:

    print(dataset.profile)

    # Read the raster data
    band = dataset.read(1).astype('float32')

    # Handle no-data values
    band[band == dataset.nodata] = np.nan

    # Normalize the data (common Sentinel-2 scaling factor: 10000)
    band /= 10000.0

    # Clip values to enhance visual contrast
    vmin, vmax = np.nanpercentile(band, (2, 98))

    plt.figure(figsize=(10, 10))
    plt.imshow(band, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Sentinel-2 Band 4 (scaled)')
    plt.colorbar(label='Reflectance')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig('/datasets/SAT_sentinel_band_4.png', dpi=300, bbox_inches='tight')
4/161:

# Open COG using rasterio
with rasterio.open(cog_url) as dataset:
    print(dataset.profile)

    # Read the raster data
    band = dataset.read(1).astype('float32')

    # Handle no-data values
    band[band == dataset.nodata] = np.nan

    # Normalize the data (common Sentinel-2 scaling factor: 10000)
    band /= 10000.0

    # Clip values to enhance visual contrast
    vmin, vmax = np.nanpercentile(band, (2, 98))

    plt.figure(figsize=(10, 10))
    plt.imshow(band, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Sentinel-2 Band 4 (scaled)')
    plt.colorbar(label='Reflectance')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig('/datasets/sentinel_band4_new.png', dpi=300, bbox_inches='tight')
4/162:

# Open COG using rasterio
with rasterio.open(local_file) as dataset:
    print(dataset.profile)

    # Read the raster data
    band = dataset.read(1).astype('float32')

    # Handle no-data values
    band[band == dataset.nodata] = np.nan

    # Normalize the data (common Sentinel-2 scaling factor: 10000)
    band /= 10000.0

    # Clip values to enhance visual contrast
    vmin, vmax = np.nanpercentile(band, (2, 98))

    plt.figure(figsize=(10, 10))
    plt.imshow(band, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('Sentinel-2 Band 4 (scaled)')
    plt.colorbar(label='Reflectance')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig('/datasets/SAT_sentinel_band4_new.png', dpi=300, bbox_inches='tight')
4/163: array
4/164: array.dims
4/165: array.dtype
4/166: array.shape
4/167: len(array.shape)
4/168: size(array)
4/169: Size(array)
4/170: array.size
4/171: array.ndim
4/172: array.ndim-1
4/173:
for i in range(array.ndim):
    print(i)
4/174:
for i in range(array.ndim):
    print(array[i])
4/175: range(array.ndim)
4/176:
for i in range(array.ndim):
    print(array.shape)
4/177:
for i in range(array.ndim):
    print(array.shape[i])
    )
4/178:
for i in range(array.ndim):
    print(array.shape[i])
4/179: array.shape
4/180: array[0,1]
4/181: array[0,:]
4/182: len(array[0,:])
4/183:
for i in range(10):
    print(array[0,:])
4/184:
for i in range(10):
    print(array[i,:])
4/185:
for i in range(3):
    print(array[i,:])
4/186:
for i in range(3):
    print(array[i,:20])
4/187:
for i in range(3):
    print(array[i,:200])
4/188:
for i in range(3):
    print(array[i,:50])
4/189:
for i in range(3):
    print(array[i,:30])
4/190:
for i in range(3):
    print(array[i,30:])
4/191:
for i in range(3):
    print(array[i,-30:])
4/192:
for i in range(3):
    print(array[i,-300:-350])
4/193:
for i in range(3):
    print(array[i,-300:-450])
4/194:
for i in range(3):
    print(array[i,-300:-200])
4/195:
for i in range(3):
    print(array[i,-300:-250])
4/196:
for i in range(3):
    print(array[i,-300:-350])
4/197:
for i in range(3):
    print(array[i,-300:-250])
4/198:
for i in range(3):
    print(array[i,-300:-270])
4/199:
for i in range(3):
    print(array[i,-400:-570])
4/200:
for i in range(3):
    print(array[i,-500:-570])
4/201:
for i in range(3):
    print(array[i,-500:-470])
4/202:
for i in range(1000,1003):
    print(array[i,-500:-470])
4/203:
for i in range(3000,3003):
    print(array[i,-3000:-3070])
4/204:
for i in range(3000,3003):
    print(array[i,-3000:-2070])
4/205:
for i in range(3000,3003):
    print(array[i,-3000:-2700])
4/206: array
4/207: array[;,;]
4/208: array[:,:]
4/209: array[:,:]
4/210: %history -f ./ipython_img_ds.py
4/211: type(array)
4/212:
for i in range(3000,3003):
    
    print(array[i,-3000:-3070])
4/213: cleaer
4/214: clear
4/215: array
4/216: gdf_dc = gpd_read_file('./tl_2024_us_state/tl_2024_us_state.shp')
4/217: ls
4/218: ls /
4/219: curl -o /datasets/tl_2024_us_state.zip https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip
4/220: curl -o /datasets/tl_2024_us_state.zip https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip
4/221: !curl -o /datasets/tl_2024_us_state.zip https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip
4/222:  unzip /datasets/tl_2024_us_state.zip -d /datasets/tl_2024_us_state
4/223: ! unzip /datasets/tl_2024_us_state.zip -d /datasets/tl_2024_us_state
4/224: gdf_dc = gpd_read_file('/datasets//tl_2024_us_state/tl_2024_us_state.shp')
4/225: gdf_dc = gpd_read_file('/datasets/tl_2024_us_state/tl_2024_us_state.shp')
4/226: gdf_dc = gpd.read_file('/datasets/tl_2024_us_state/tl_2024_us_state.shp')
4/227: import geopandas as gpd
4/228: gdf_dc = gpd.read_file('/datasets/tl_2024_us_state/tl_2024_us_state.shp')
4/229: gdf_dc
4/230: gdf_dc.head
4/231: ()
4/232: gdf_dc.head()
4/233: gdf_dc.items
4/234: gdf_dc.columns
4/235: gdf_dc.iloc[0]
4/236: gdf_dc.iloc[1]
4/237: gdf_dc.iloc[2]
4/238: len(gdf)
4/239: len(gdf_dc)
4/240: gdf_dc.iloc[2]
4/241: gdf_dc[gdf_dc.STUSPS=='DC']
4/242: gdf_temp = gdf_dc
4/243: gdf_dc = gdf_temp[gdf_temp.STUSPS=='DC']
4/244: gdf_dc
4/245: gdf_temp
4/246: gdf_us = gdf_temp
4/247: remove(gdf_temp)
4/248: del gdf_temp
4/249: gdf_temp
4/250: clear
4/251: gdf_dc.bounds
4/252: type(gdf_dc.bounds)
4/253: type(gdf_dc.bounds).values()
4/254: type(gdf_dc.bounds).values())
4/255: type(gdf_dc.bounds).values)
4/256: gdf_dc.bounds.values
4/257: gdf_dc.bounds.values.to_list()
4/258: gdf_dc.bounds.values.tolist()
4/259: list_dc_bounds = gdf_dc.bounds.values.tolist()
4/260: list_dc_bounds
4/261: params = {}
4/262: params['limit'] = 400
4/263: params['bbox'] = tuple(list_dc_bounds)
4/264: params
4/265: date_time = "2024-09-01T00:00:00Z/2024-10-31T23:59:59Z"
4/266: params['datetime'] = date_time
4/267: collections = ['landsat-c2l2-sr']
4/268: params['collections'] = collections
4/269:
params['query'] = {"platform": {"in": ["LANDSAT_9"]},
                   'eo:cloud_cover':{'lte': 20}}
4/270: params
4/271: import requests as rq
4/272:
def fetch_stac_server(query):
    '''
    Queries the stac-server (STAC) backend.
    query is a python dictionary to pass as json to the request.
    '''
    
    search_url = f"https://landsatlook.usgs.gov/stac-server/search"
    query_return = rq.post(search_url, json=query).json()
    error = query_return.get("message", "")
    if error:
        raise Exception(f"STAC-Server failed and returned: {error}")
        
    print(f"Items Found: {len(query_return['features'])}")    
    
    for q in query_return['features']: print(f"Platform: {q['properties']['platform']}, Cloud Cover: {q['properties']['eo:cloud_cover']}, Collection: {q['description']}, ID: {q['id']}")
        
    return query_return
4/273: query_return = fetch_stac_server(params)
4/274: query_return
4/275: search_url = f"https://landsatlook.usgs.gov/stac-server/search"
4/276: query_return = rq.post(search_url, json=params).json()
4/277: query_return
4/278: list_dc_bounds
4/279: params['bbox'] = tuple(list_dc_bounds[0])
4/280: query_return = rq.post(search_url, json=params).json()
4/281: query_return
4/282:
def fetch_stac_server(query):
    '''
    Queries the stac-server (STAC) backend.
    query is a python dictionary to pass as json to the request.
    '''
    
    search_url = f"https://landsatlook.usgs.gov/stac-server/search"
    query_return = rq.post(search_url, json=query).json()
    error = query_return.get("message", "")
    if error:
        raise Exception(f"STAC-Server failed and returned: {error}")
        
    print(f"Items Found: {len(query_return['features'])}")    
    
    for q in query_return['features']: print(f"Platform: {q['properties']['platform']}, Cloud Cover: {q['properties']['eo:cloud_cover']}, Collection: {q['description']}, ID: {q['id']}")
        
    return query_return
4/283: query_return = fetch_stac_server(params)
4/284: query_return
4/285: query_return = fetch_stac_server(params)
4/286:
#Function to list query assets from select feature (item)
def list_item_assets(query_item):
    print(query_item['id'])
    
    # Print some of the  metadata attributes for the product
    print(f"> Product Family: {query_item['description']}")
    print(f"> Product ID: {query_item['id']}")
    print(f"> Acquisition Date: {query_item['properties']['datetime']}")
    print(f"> Spacecraft (Platform): {query_item['properties']['platform']}")
    print(f"> Cloud Cover: {query_item['properties']['eo:cloud_cover']}%.")
    print(f"> Number of assets: {len(query_item['assets'])} assets")
    print(f"> BBOX: {query_item['bbox']}") 
    
    for a in query_item['assets']: #print assets in each feature
        
        print('\n'+ a)
        
        try:
            print(f"Asset: {query_item['assets'][a]['title']}")
            print(f"type: {query_item['assets'][a]['type']}")
            print(f"Description: {query_item['assets'][a]['description']}")
            print(f"Role: {query_item['assets'][a]['roles']}")
            print(f"S3 URL: {query_item['assets'][a]['alternate']['s3']['href']}")
            
            # for asset in query_item['assets']:
            #     print(f"{asset}: {query_item['assets'][asset]['title']}")

        except:
            print(f"Role: {query_item['assets'][a]['roles']}")
            print(f"URL: {query_item['assets'][a]['href']}")

            continue
            
    return 0
4/287:
query_item = query_return['features'][0] #search a single feature (item) in the collection
list_item_assets(query_item)
4/288: query_item
4/289: type(query_item)
4/290: query_item.keys()
4/291: [print(key) for key in query_item.keys()]
4/292:
bands = ['red', 'green', 'blue']
band_ids = []
band_links = []

for b in bands: #sort through a string to find relevant band names
        band_links.append(query_item['assets'][b]['alternate']['s3']['href'])
        band_ids.append(query_item['assets'][b]['eo:bands'][0]['name'])

print(band_ids)
4/293: band_links
4/294: query_return['features']
4/295: query_return.keys
4/296: query_return.keys()
4/297: len(query_return['features'])
4/298: query_return['features']
4/299: type(query_return['features'])
4/300: query_return['features'][0]
4/301: query_return['features'][1]
4/302: query_return['features'][0]
4/303: query_return['features'][1]
4/304: query_return['features'][1]
4/305: query_return = fetch_stac_server(params)
4/306:
query_item = query_return['features'][0] #search a single feature (item) in the collection
list_item_assets(query_item)
4/307:
bands = ['red', 'green', 'blue']
band_ids = []
band_links = []

for b in bands: #sort through a string to find relevant band names
        band_links.append(query_item['assets'][b]['alternate']['s3']['href'])
        band_ids.append(query_item['assets'][b]['eo:bands'][0]['name'])

print(band_ids)
4/308: band_links
4/309:
import requests
import rasterio as rio

url = 's3://usgs-landsat/collection02/level-2/standard/oli-tirs/2024/015/033/LC09_L2SP_015033_20241020_20241022_02_T1/LC09_L2SP_015033_20241020_20241022_02_T1_SR_B4.TIF'

local_file = 'DC_4.tif'

# Robustly download entire file
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(local_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# Then open locally
with rio.open(local_file) as src:
    cog_dc = src.read(1)
    print("Filename:", src.name)
    print("Driver:", src.driver)
    print("Width, Height:", src.width, src.height)
    print("CRS:", src.crs)
    print("Transform (georeferencing):", src.transform)
    print("Number of Bands:", src.count)
    print("Data type:", src.dtypes)
    print("Bounds:", src.bounds)
    print("Compression:", src.profile.get("compress"))
print(cog_dc.shape)
4/310: aws_session = AWSSession(boto3.Session(), requester_pays=True)
4/311:
def retrieve_cog(geotiff_path, aoi_geodf):
    """
    Retrieve a Cloud-Optimized GeoTIFF (COG) within a specified bounding box.

    Parameters:
        - geotiff_path (str): Path to the COG GeoTIFF file.
        - aoi_geodf (geopandas.GeoDataFrame): GeoDataFrame representing the area of interest.

    Returns:
        - cog (numpy.ndarray): Numpy array containing the data within the specified bounding box.
    """
    with rio.Env(aws_session):
        with rio.open(geotiff_path) as src:
            # Assuming aoi_geodf has a different CRS, transform it to match the CRS of the raster
            aoi_geodf = aoi_geodf.to_crs(src.crs)
            cog = src.read(1, window=from_bounds(
                aoi_geodf.bounds.minx[0],
                aoi_geodf.bounds.miny[0],
                aoi_geodf.bounds.maxx[0],
                aoi_geodf.bounds.maxy[0],
                src.transform))
    return cog
4/312:
plt.figure(figsize=(5, 5))
rio_show(retrieve_cog(band_links[0],aoi_geodf), cmap='viridis')
plt.show()
4/313: from rasterio.plot import show as rio_show
4/314:
plt.figure(figsize=(5, 5))
rio_show(retrieve_cog(band_links[0],aoi_geodf), cmap='viridis')
plt.show()
4/315:
plt.figure(figsize=(5, 5))
rio_show(retrieve_cog(band_links[0],gdf_dc), cmap='viridis')
plt.show()
4/316: aws_session
4/317:
plt.figure(figsize=(5, 5))
rio_show(retrieve_cog(band_links[0],gdf_dc), cmap='viridis')
plt.show()
4/318:
def retrieve_cog(geotiff_path, aoi_geodf):
    """
    Retrieve a Cloud-Optimized GeoTIFF (COG) within a specified bounding box.

    Parameters:
        - geotiff_path (str): Path to the COG GeoTIFF file.
        - aoi_geodf (geopandas.GeoDataFrame): GeoDataFrame representing the area of interest.

    Returns:
        - cog (numpy.ndarray): Numpy array containing the data within the specified bounding box.
    """
    # with rio.Env(aws_session):
        with rio.open(geotiff_path) as src:
            # Assuming aoi_geodf has a different CRS, transform it to match the CRS of the raster
            aoi_geodf = aoi_geodf.to_crs(src.crs)
            cog = src.read(1, window=from_bounds(
                aoi_geodf.bounds.minx[0],
                aoi_geodf.bounds.miny[0],
                aoi_geodf.bounds.maxx[0],
                aoi_geodf.bounds.maxy[0],
                src.transform))
    return cog
4/319:
def retrieve_cog(geotiff_path, aoi_geodf):
    """
    Retrieve a Cloud-Optimized GeoTIFF (COG) within a specified bounding box.

    Parameters:
        - geotiff_path (str): Path to the COG GeoTIFF file.
        - aoi_geodf (geopandas.GeoDataFrame): GeoDataFrame representing the area of interest.

    Returns:
        - cog (numpy.ndarray): Numpy array containing the data within the specified bounding box.
    """
    # with rio.Env(aws_session):
    with rio.open(geotiff_path) as src:
        # Assuming aoi_geodf has a different CRS, transform it to match the CRS of the raster
        aoi_geodf = aoi_geodf.to_crs(src.crs)
        cog = src.read(1, window=from_bounds(
            aoi_geodf.bounds.minx[0],
            aoi_geodf.bounds.miny[0],
            aoi_geodf.bounds.maxx[0],
            aoi_geodf.bounds.maxy[0],
            src.transform))
    return cog
4/320:
plt.figure(figsize=(5, 5))
rio_show(retrieve_cog(band_links[0],gdf_dc), cmap='viridis')
plt.show()
4/321: params
4/322: geotiff_path
 5/1: geotiff_path
 5/2:
import rasterio as rio
from rasterio.env import Env

url = '/vsicurl/https://landsat-pds.s3.amazonaws.com/c1/L8/042/034/LC08_L1TP_042034_20170616_20170629_01_T1/LC08_L1TP_042034_20170616_20170629_01_T1_B4.TIF'

with Env(AWS_NO_SIGN_REQUEST='YES'):
    with rio.open(url) as src:
        cog = src.read(1)

print(cog.shape)
 5/3: %history -f ./ipython_img_ds.py
 5/4: ls
 5/5: ls -la ipython_img_ds.py
 5/6: ls -la ipython_img_ds.py
 5/7: import rasterio as rio
 5/8: rio.open('s3://usgs-landsat/collection02/level-2/standard/oli-tirs/2024/015/033/LC09_L2SP_015033_20241020_20241022_02_T1/LC09_L2SP_015033_20241020_20241022_02_T1_SR_B4.TIF')
 5/9:
import rasterio

# Define the S3 URL
s3_url = "s3://usgs-landsat/collection02/level-2/standard/oli-tirs/2024/015/033/LC09_L2SP_015033_20241020_20241022_02_T1/LC09_L2SP_015033_20241020_20241022_02_T1_SR_B4.TIF"

# Open the file using rasterio
with rasterio.open(s3_url) as src:
    print(f"CRS: {src.crs}")
    print(f"Bounds: {src.bounds}")
    print(f"Shape: {src.shape}")
    # Read the data
    band_data = src.read(1)  # Read the first band (Band 4 in this case)

# Do whatever processing you need with the band data
5/10:
import os
import rasterio

# Set environment variable to disable AWS signing (public access)
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# Define the S3 URL
s3_url = "s3://usgs-landsat/collection02/level-2/standard/oli-tirs/2024/015/033/LC09_L2SP_015033_20241020_20241022_02_T1/LC09_L2SP_015033_20241020_20241022_02_T1_SR_B4.TIF"

# Open the file using rasterio
with rasterio.open(s3_url) as src:
    print(f"CRS: {src.crs}")
    print(f"Bounds: {src.bounds}")
    print(f"Shape: {src.shape}")
    # Read the data
    band_data = src.read(1)  # Read the first band (Band 4 in this case)

# Do whatever processing you need with the band data
5/11:
import os
import rasterio

# Set environment variable to disable AWS signing (public access)
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# Define the S3 URL
s3_url = "s3://usgs-landsat/collection02/level-2/standard/oli-tirs/2024/015/033/LC09_L2SP_015033_20241020_20241022_02_T1/LC09_L2SP_015033_20241020_20241022_02_T1_SR_B4.TIF"

# Open the file using rasterio
with rasterio.open(s3_url) as src:
    print(f"CRS: {src.crs}")
    print(f"Bounds: {src.bounds}")
    print(f"Shape: {src.shape}")
    # Read the data
    band_data = src.read(1)  # Read the first band (Band 4 in this case)

# Do whatever processing you need with the band data
5/12:
import boto3
import rasterio
from botocore.exceptions import NoCredentialsError

# Initialize boto3 client for S3
s3_client = boto3.client('s3')

# Generate the pre-signed URL
url = s3_client.generate_presigned_url('get_object',
                                       Params={'Bucket': 'usgs-landsat',
                                               'Key': 'collection02/level-2/standard/oli-tirs/2024/015/033/LC09_L2SP_015033_20241020_20241022_02_T1/LC09_L2SP_015033_20241020_20241022_02_T1_SR_B4.TIF'},
                                       ExpiresIn=3600)  # URL expires in 1 hour

# Open the file using rasterio and the pre-signed URL
with rasterio.open(url) as src:
    print(f"CRS: {src.crs}")
    print(f"Bounds: {src.bounds}")
    print(f"Shape: {src.shape}")
    band_data = src.read(1)  # Read the first band (Band 4)
5/13:
import rasterio

# Define the S3 URL
s3_url = "https://storage.googleapis.com/gcp-public-data-landsat/collection02/level-2/standard/oli-tirs/2024/015/033/LC09_L2SP_015033_20241020_20241022_02_T1/LC09_L2SP_015033_20241020_20241022_02_T1_SR_B4.TIF
"

# Open the file using rasterio
with rasterio.open(s3_url) as src:
    print(f"CRS: {src.crs}")
    print(f"Bounds: {src.bounds}")
    print(f"Shape: {src.shape}")
    # Read the data
    band_data_dc_4 = src.read(1)  # Read the first band (Band 4 in this case)

# Do whatever processing you need with the band data
5/14:
import rasterio

# Define the S3 URL
s3_url = "https://storage.googleapis.com/gcp-public-data-landsat/collection02/level-2/standard/oli-tirs/2024/015/033/LC09_L2SP_015033_20241020_20241022_02_T1/LC09_L2SP_015033_20241020_20241022_02_T1_SR_B4.TIF"

# Open the file using rasterio
with rasterio.open(s3_url) as src:
    print(f"CRS: {src.crs}")
    print(f"Bounds: {src.bounds}")
    print(f"Shape: {src.shape}")
    # Read the data
    band_data_dc_4 = src.read(1)  # Read the first band (Band 4 in this case)

# Do whatever processing you need with the band data
 6/1: ls -la ipython_img_ds.py
 6/2: %history -f ./ipython_img_ds.py
 7/1: docker ps -a
 7/2: clear
   1: %history -f ./ipython_img_ds.py
   2: %history -g -f ./ipython_img_ds.py
