import urllib.request
from tqdm import tqdm

img_name = [
                "a1665-jn_20080821_NYC_006",
                "a2647-jmac_DSC1283",
                "a3351-KE_-1722",
                "a3661-WP_CRW_0405",
                "a3886-_DGW6415"
            ]

print('download reference images')
for name in tqdm(img_name):
    url = f"https://data.csail.mit.edu/graphics/fivek/img/tiff16_b/{name}.tif"
    save_path = f"./samples/{name}.tif"

    urllib.request.urlretrieve(url, save_path)
    print(f"Downloaded: {save_path}")

print('download gt image')
name = "a1535-kme_501"
url = f"https://data.csail.mit.edu/graphics/fivek/img/tiff16_a/{name}.tif"
save_path = f"./samples/gt_{name}.tif"
urllib.request.urlretrieve(url, save_path)
print(f"Downloaded: {save_path}")

# https://data.csail.mit.edu/graphics/fivek/img/tiff16_a/a1535-kme_501.tif
