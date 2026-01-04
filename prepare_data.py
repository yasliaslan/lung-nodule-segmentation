# Save as: prepare_data.py
import os
import argparse
import xml.etree.ElementTree as ET
from glob import glob
import numpy as np
import pydicom
from skimage.draw import polygon
from tqdm import tqdm
from PIL import Image
import random

def get_hu_image(dcm):
    img = dcm.pixel_array.astype(np.int16)
    if 'RescaleSlope' in dcm and 'RescaleIntercept' in dcm:
        img = img * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
    return img

def window_image(img, level=-600, width=1500):
    lower = level - width // 2
    upper = level + width // 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)
    img = (img * 255).astype(np.uint8)
    return img

def get_nodule_polygons(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'lidc': 'http://www.nih.gov'}
    nodules = []
    for session in root.findall(".//lidc:readingSession", ns):
        for nodule in session.findall(".//lidc:unblindedReadNodule", ns):
            for roi in nodule.findall(".//lidc:roi", ns):
                inclusion = roi.find("lidc:inclusion", ns)
                if inclusion is not None and inclusion.text.strip().upper() != "TRUE":
                    continue
                sop_uid_el = roi.find("lidc:imageSOP_UID", ns)
                if sop_uid_el is None: continue
                sop_uid = sop_uid_el.text.strip()
                coords = []
                for edge in roi.findall("lidc:edgeMap", ns):
                    x = int(edge.find("lidc:xCoord", ns).text)
                    y = int(edge.find("lidc:yCoord", ns).text)
                    coords.append((y, x))
                if len(coords) >= 3:
                    nodules.append({'sop_uid': sop_uid, 'coords': coords})
    return nodules

def load_dicom_slices(dicom_files):
    slices = {}
    ordered = []
    for path in dicom_files:
        try:
            dcm = pydicom.dcmread(path)
            if dcm.Modality != "CT": continue
            sop_uid = dcm.SOPInstanceUID
            slices[sop_uid] = dcm
            ordered.append((float(dcm.ImagePositionPatient[2]), sop_uid))
        except: continue
    ordered.sort(key=lambda x: x[0])
    return slices, [uid for _, uid in ordered]

def create_mask_from_polygons(nodule_data, slices_dict, slice_order):
    if not slice_order: return []
    sample_dcm = slices_dict[slice_order[0]]
    image_shape = sample_dcm.pixel_array.shape
    mask = np.zeros((len(slice_order), *image_shape), dtype=np.uint8)
    uid_to_idx = {uid: idx for idx, uid in enumerate(slice_order)}
    for nod in nodule_data:
        sop_uid = nod['sop_uid']
        if sop_uid not in uid_to_idx: continue
        z_idx = uid_to_idx[sop_uid]
        poly = nod['coords']
        rr, cc = polygon([p[0] for p in poly], [p[1] for p in poly], shape=image_shape)
        mask[z_idx][rr, cc] = 1
    return mask

def main(lidc_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_path, split), exist_ok=True)

    patients = sorted(os.listdir(lidc_path))
    random.seed(42)
    random.shuffle(patients)
    n_total = len(patients)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)

    print(f"Total Patients: {n_total} | Train: {n_train} | Val: {n_val} | Test: {n_total - n_train - n_val}")

    for i, pid in enumerate(tqdm(patients)):
        patient_path = os.path.join(lidc_path, pid)
        dicom_files = glob(os.path.join(patient_path, "**", "*.dcm"), recursive=True)
        xml_files = glob(os.path.join(patient_path, "**", "*.xml"), recursive=True)

        if not dicom_files: continue
        slices_dict, slice_order = load_dicom_slices(dicom_files)
        if not slice_order: continue

        nodules = []
        if xml_files:
            nodules = get_nodule_polygons(xml_files[0])
        
        shape = slices_dict[slice_order[0]].pixel_array.shape
        mask = create_mask_from_polygons(nodules, slices_dict, slice_order) if nodules else np.zeros((len(slice_order), shape[0], shape[1]), dtype=np.uint8)
        
        images = [window_image(get_hu_image(slices_dict[uid])) for uid in slice_order]

        if i < n_train: split = 'train'
        elif i < n_train + n_val: split = 'val'
        else: split = 'test'

        patient_out_dir = os.path.join(output_path, split, pid)
        os.makedirs(os.path.join(patient_out_dir, "slices"), exist_ok=True)
        os.makedirs(os.path.join(patient_out_dir, "masks"), exist_ok=True)

        for z in range(len(slice_order)):
            Image.fromarray(images[z]).save(os.path.join(patient_out_dir, "slices", f"slice_{z:03d}.png"))
            Image.fromarray(mask[z] * 255).save(os.path.join(patient_out_dir, "masks", f"mask_{z:03d}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lidc_path", type=str, required=True, help="Path to LIDC-IDRI root folder")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Output path for PNGs")
    args = parser.parse_args()
    main(args.lidc_path, args.output_path)