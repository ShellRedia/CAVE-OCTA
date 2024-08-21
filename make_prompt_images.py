import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from itertools import product

def get_certer_coords(mask):
    structure = ndimage.generate_binary_structure(2, 2)
    labelmaps, connected_num = ndimage.label(mask, structure=structure)
    coord_lst = []
    for mask_i in range(1, connected_num+1):
        rows, cols = np.where(labelmaps == mask_i)
        coord_lst.append(tuple(map(lambda x:int(x.mean()), (rows, cols))))
    return coord_lst

def make_concat_labels(save_dir="special_points", label_type="Artery", fov="3M"):
    sample_ids = {"3M":range(10301, 10501), "6M":range(10001, 10301)}[fov]
    image_size = {"3M":304, "6M":400}[fov]

    save_dir = "{}/{}/{}".format(save_dir, fov, label_type)
    os.makedirs(save_dir, exist_ok=True)

    batch_size = 10
    point_size = 2
    cols = 5
    rows = batch_size // cols

    padding = 20

    for i, sample_id in tqdm(list(enumerate(sample_ids))):
        batch_i = i % batch_size
        if batch_i == 0: 
            batch_shape = ((image_size+padding) * rows, (image_size+padding) * cols, 3)
            batch_image = np.zeros(batch_shape)
            batch_image[:,:] = (127, 127, 127)

        image_dir = "datasets/OCTA-500/3M/GT_{}".format(label_type)

        image = cv2.imread("{}/{}.bmp".format(image_dir, sample_id), cv2.IMREAD_COLOR)

        center_mask = image.copy()
        center_mask[1:-1, 1:-1] = 0

        bifurcation_dir = "datasets/OCTA-500/{}/GT_Point/{}/Bifurcation".format(fov, label_type)
        endpoints_dir = "datasets/OCTA-500/{}/GT_Point/{}/Endpoint".format(fov, label_type)

        bifurcations = np.load("{}/{}.npy".format(bifurcation_dir, sample_id))
        endpoints = np.load("{}/{}.npy".format(endpoints_dir, sample_id))

        for y, x in bifurcations: 
            cv2.circle(image, (x, y), point_size, (0, 255, 0), -1)

        for y, x in endpoints: 
            cv2.circle(image, (x, y), point_size, (0, 0, 255), -1)
        
        for y, x in get_certer_coords(center_mask[:,:,0]): 
            cv2.circle(image, (x, y), point_size, (0, 0, 255), -1)

        cv2.imwrite("{}/{}.png".format(save_dir, sample_id), image)

        
        r, c = (batch_i // cols) * (image_size+padding), (batch_i % cols) * (image_size+padding)
        batch_image[r:r+image_size,  c:c+image_size] = image

        if batch_i == batch_size-1:  cv2.imwrite("{}/batch_{:0>3}.png".format(save_dir, i // batch_size), batch_image)

def extract_coords_from_concat_labels(save_dir="modified_points", label_type = "Artery", fov = "6M"):
    sample_ids = {"3M":range(10301, 10501), "6M":range(10001, 10301)}[fov]
    image_size = {"3M":304, "6M":400}[fov]

    batch_size = 10
    cols = 5
    rows = batch_size // cols

    padding = 20

    annotated_dir = "special_points-copy/{}/{}".format(fov, label_type)

    save_dir = "{}/{}/{}".format(save_dir, fov, label_type)
    os.makedirs("{}/{}".format(save_dir, "Bifurcation"), exist_ok=True)
    os.makedirs("{}/{}".format(save_dir, "Endpoint"), exist_ok=True)

    for batch_i in tqdm(list(range(len(sample_ids) // batch_size))):
        annotated_image = cv2.imread("{}/batch_{:0>3}.png".format(annotated_dir, batch_i), cv2.IMREAD_COLOR)

        for r, c in product(range(rows), range(cols)):
            sample_id = sample_ids[0] + batch_i * batch_size + r * cols + c
            r, c = r * (image_size + padding), c * (image_size + padding)
            single_image = annotated_image[r:r+image_size,  c:c+image_size]

            diff_map = np.abs(single_image - (0, 255, 0))
            diff_map = np.sum(diff_map, axis=2)
            bifurcations_mask = np.where(diff_map < 5, 1, 0)
            coords_bi = get_certer_coords(bifurcations_mask)
            

            diff_map = np.abs(single_image - (0, 0, 255))
            diff_map = np.sum(diff_map, axis=2)
            endpoints_mask = np.where(diff_map < 5, 1, 0)
            coords_end = get_certer_coords(endpoints_mask)

            np.save("{}/{}/{}.npy".format(save_dir, "Bifurcation", sample_id), coords_bi)
            np.save("{}/{}/{}.npy".format(save_dir, "Endpoint", sample_id), coords_end)

def extract_from_sparse_annotation(extracted_dir="SparseAnnotation", save_dir="special_points", label_type="Vein"):
    fov2image_size = {"3M":304, "6M":400}

    sample_id2fov = {}
    for sample_id in range(10301, 10501): sample_id2fov[str(sample_id)] = "3M"
    for sample_id in range(10001, 10301): sample_id2fov[str(sample_id)] = "6M"

    bifurcation_dir = "{}/{}/Bifurcation/predictions".format(extracted_dir, label_type)
    endpoint_dir = "{}/{}/Endpoint/predictions".format(extracted_dir, label_type)

    
    sample_ids = sorted([x[:-4] for x in os.listdir(bifurcation_dir)])

    save_dir = "{}/{}".format(save_dir, label_type)

    os.makedirs(save_dir, exist_ok=True)

    batch_size = 10
    point_size = 2
    cols = 5
    rows = batch_size // cols

    padding = 20

    for i, sample_id in tqdm(list(enumerate(sample_ids))):
        bifurcation = cv2.imread("{}/{}.png".format(bifurcation_dir, sample_id), cv2.IMREAD_COLOR)
        endpoint = cv2.imread("{}/{}.png".format(endpoint_dir, sample_id), cv2.IMREAD_COLOR)

        fov = sample_id2fov[sample_id]
        image_size = fov2image_size[fov]
        h, w, _ = bifurcation.shape

        bifurcation = bifurcation[:,w//2:][:,:,0]
        endpoint = endpoint[:,w//2:][:,:,0]

        kernel = np.ones((5, 5),np.uint8)

        # 执行腐蚀操作
        bifurcation = cv2.erode(bifurcation, kernel, iterations=1)
        endpoint = cv2.erode(endpoint, kernel, iterations=1)

        label = cv2.imread("datasets/OCTA-500/{}/GT_{}/{}.bmp".format(fov, label_type, sample_id), cv2.IMREAD_COLOR)

        for y, x in get_certer_coords(bifurcation):
            x, y = int(x * image_size // h), int(y * image_size // h)
            cv2.circle(label, (x, y), point_size, (0, 255, 0), -1)
        for y, x in get_certer_coords(endpoint):
            x, y = int(x * image_size // h), int(y * image_size // h)
            cv2.circle(label, (x, y), point_size, (0, 0, 255), -1)

        batch_i = i % batch_size
        if batch_i == 0: 
            batch_shape = ((image_size+padding) * rows, (image_size+padding) * cols, 3)
            batch_image = np.zeros(batch_shape)
            batch_image[:,:] = (127, 127, 127)
        
        r, c = (batch_i // cols) * (image_size+padding), (batch_i % cols) * (image_size+padding)
        batch_image[r:r+image_size,  c:c+image_size] = label

        if batch_i == batch_size-1:  cv2.imwrite("{}/batch_{:0>3}.png".format(save_dir, i // batch_size), batch_image)


if __name__=="__main__":
    pass
    # extract_from_sparse_annotation()

    # for label_type, fov in product(["Artery", "Vein"], ["3M", "6M"]):
    #     print(label_type, fov)
    #     extract_coords_from_concat_labels(label_type=label_type, fov=fov)

