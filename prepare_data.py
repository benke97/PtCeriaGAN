import pandas as pd
import pickle as pkl
import os
import numpy as np
import mrcfile
import random
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from math import inf
import tifffile as tiff
import glob

def plot_im_with_hist(image):
    #plot image and histogram in a 1x2 subfigure:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the image on the first subplot
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')  # Hide axes ticks
    
    # Plot the histogram on the second subplot
    axs[1].hist(image.ravel(), bins=100, range=[np.min(image),np.max(image)], fc='k', ec='k')
    axs[1].set_title('Histogram')
    
    # Display the plot
    plt.show()

def add_noise(clean_dir, save_dir):
    #files
    file_names = os.listdir(clean_dir)
    file_paths = [os.path.join(clean_dir,f"{file_name}") for file_name in file_names]
    save_paths = [os.path.join(save_dir, f"{file_name}") for file_name in file_names]
    #dataframe_paths = [os.path.join(dataframe_dir, f"structure_{file_name.split('_')[0].split('.')[0]}.pkl") for file_name in file_names]
    for file_path,save_path in zip(file_paths,save_paths):
        
        image = np.load(file_path)['arr_0']

        #Add gaussian noise
        gaussian_noise = np.random.normal(0, 0.0001, image.shape)
        noisy_image = np.clip(image+gaussian_noise,0,np.max(image+gaussian_noise))

        #Apply Poisson noise
        noisy_image = np.random.poisson(noisy_image*500)/500
        noisy_image = gaussian_filter(noisy_image, sigma=0.5)

        #Save noisy image
        np.savez_compressed(save_path, noisy_image)

def convert_exp_to_npz(raw_dir, save_dir):
        
        def convert_tiff_to_npz(tiff_path, npz_path):
            image = tiff.imread(tiff_path)
            np.savez_compressed(npz_path, image)
    
        file_names = os.listdir(raw_dir)
        file_paths = [os.path.join(raw_dir,f"{file_name}") for file_name in file_names]
        save_paths = [os.path.join(save_dir, f"{file_name.split('.')[0]}.npz") for file_name in file_names]
        
        for file_path,save_path in zip(file_paths,save_paths):
            convert_tiff_to_npz(file_path,save_path)

def convert_raw_to_npz(raw_dir, save_dir, apply_probe=False,dataframe_dir = "data/noiser/dataframes/"):

    def convert_mrc_to_npz(mrc_path, npz_path, apply_probe=False, structure_df = None):
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            img_data = mrc.data
            if apply_probe:
                pixel_size = structure_df['pixel_size'].iloc[0]
                hwhm_nm = 0.05  # HWHM in nanometers
                hwhm_pixels = hwhm_nm / pixel_size
                sigma = hwhm_pixels / np.sqrt(2 * np.log(2))
                probe = sigma
                img_data = gaussian_filter(img_data, sigma=probe)
            np.savez_compressed(npz_path, img_data)    




    file_names = os.listdir(raw_dir)
    file_paths = [os.path.join(raw_dir,f"{file_name}") for file_name in file_names]
    save_paths = [os.path.join(save_dir, f"{file_name.split('_')[0]}.npz") for file_name in file_names]
    
    for file_name, file_path, save_path in zip(file_names, file_paths, save_paths):
        idx = file_name.split('_')[0]

        with open(os.path.join(dataframe_dir,f"structure_{idx}.pkl"), "rb") as f:
            structure_df = pkl.load(f)

        convert_mrc_to_npz(file_path,save_path,apply_probe=True, structure_df=structure_df)

def normalize(data_dir, normalization="minmax"):
    #Assumes data_dir is filled with only .npz files

    def min_max_normalize(image,min,max):
        return (image-min)/(max-min)

    if normalization != "minmax":
        NotImplementedError
    files = os.listdir(data_dir)
    global_max = -inf
    global_min = inf
    for file in files:
        
        image = np.load(os.path.join(data_dir,file))["arr_0"]
        max_val = np.max(image)
        min_val = np.min(image)
        
        if max_val > global_max:
            global_max = max_val
        if min_val < global_min:
            global_min = min_val
        
    for file in files: 
        image = np.load(os.path.join(data_dir,file))["arr_0"]
        image = min_max_normalize(image,global_min,global_max)
        np.savez_compressed(os.path.join(data_dir,file), image)  

def normalize_exp(data_dir, normalization="minmax"):
    #Assumes data_dir is filled with only .npz files
    
    def min_max_normalize(image,min,max):
        return (image-min)/(max-min)
    
    files = os.listdir(data_dir)
    global_max = -inf
    for file in files:
        
        image = np.load(os.path.join(data_dir,file))["arr_0"]
        image = image-(np.min(image)+300)#Get rid of the background
        
        #set negative values to 0
        image[image<=0] = 0

        max_val = np.max(image)
        
        if max_val > global_max:
            global_max = max_val

    for file in files: 
        image = np.load(os.path.join(data_dir,file))["arr_0"]
        image = image-(np.min(image)+300) #Get rid of the background
        image = min_max_normalize(image,0,global_max)
        image[image<=0] = 0
        np.savez_compressed(os.path.join(data_dir,file), image)  

def split_and_save(split_ratio, data_dir, domain_name, return_split=False, split="random"):
    #Assumes data_dir is filled with only .npz files
    files = os.listdir(sim_dir)
    num_train = int(len(files)*split_ratio)
    num_val = len(files) - num_train
    if split == "random":
        train_files = random.sample(files, num_train)
        val_files = [file for file in files if file not in train_files]
    else:
        train_files = split[0]
        val_files = split[1]
    
    # navigate back one directory
    root = os.path.dirname(data_dir.rstrip('/'))
    root = os.path.join(root, '')

    train_dir = os.path.join(root, "train", domain_name)
    val_dir = os.path.join(root, "val", domain_name)
    for file in train_files:
        image = np.load(os.path.join(data_dir,file))["arr_0"]
        np.savez_compressed(os.path.join(train_dir,file), image)
    
    for file in val_files:
        image = np.load(os.path.join(data_dir,file))["arr_0"]
        np.savez_compressed(os.path.join(val_dir,file), image)
    
    if return_split:
        return train_files, val_files

def split_and_save_exp(exp_dir, sim_dir, split=0.8, train_path="data/noiser/train/exp", val_path="data/noiser/val/exp"):
    with open("data/exp_info.pkl", "rb") as f:
        exp_info = pkl.load(f)
    
    tot_sim_images = len(os.listdir(sim_dir))
    train_sim_images = int(tot_sim_images * split)
    val_sim_images = tot_sim_images - train_sim_images

    particle_to_files = {}
    for unique_particle_id in exp_info.unique_particle_id.unique():
        particle_rows = exp_info[exp_info.unique_particle_id == unique_particle_id]
        all_files = []
        for idx, row in particle_rows.iterrows():
            all_files += row.file_list
        particle_to_files[unique_particle_id] = all_files

    num_files_per_particle = [len(files) for files in particle_to_files.values()]

    num_exp_files = len(os.listdir(exp_dir))
    num_train = int(num_exp_files * split)

    # Ensure all series with columns_visible=1 are included in the training set
    visible_files = []
    for idx, row in exp_info.iterrows():
        if row.columns_visible == 1:
            visible_files += row.file_list
    
    i = len(visible_files)
    selection_train = np.zeros_like(num_files_per_particle)
    for idx, (unique_particle_id, files) in enumerate(particle_to_files.items()):
        if any(file in visible_files for file in files):
            selection_train[idx] = 1

    while i < num_train:
        idx = random.choice(range(len(num_files_per_particle)))
        if selection_train[idx] == 0:
            selection_train[idx] = 1
            i += num_files_per_particle[idx]

    num_train = i

    # Create lists of all selected files
    train_files = []
    val_files = []
    for idx, (unique_particle_id, files) in enumerate(particle_to_files.items()):
        if selection_train[idx] == 1:
            train_files += files
        else:
            val_files += files
    
    exp_info["split"] = None
    for idx, row in exp_info.iterrows():
        if row.file_list[0] in train_files:
            exp_info.at[idx, "split"] = 1
        else:
            exp_info.at[idx, "split"] = 0
    
    with open("data/exp_info.pkl", "wb") as f:
        pkl.dump(exp_info, f)

    # Shuffle files in train_files and val_files
    random.shuffle(train_files)
    random.shuffle(val_files)
    assert len(train_files) > train_sim_images
    assert len(val_files) > val_sim_images
    train_files = random.sample(train_files, train_sim_images)
    val_files = random.sample(val_files, val_sim_images)

    for file in train_files:
        image = np.load(os.path.join(exp_dir, file))["arr_0"]
        np.savez_compressed(os.path.join(train_path, file), image)

    for file in val_files:
        image = np.load(os.path.join(exp_dir, file))["arr_0"]
        np.savez_compressed(os.path.join(val_path, file), image)

def delete_all_files_in_directories(directories):
    for directory in directories:
        files = glob.glob(os.path.join(directory, '*'))
        for file in files:
            try:
                if os.path.isfile(file) or os.path.islink(file):
                    os.unlink(file)
                elif os.path.isdir(file):
                    os.rmdir(file)
            except Exception as e:
                print(f'Failed to delete {file}. Reason: {e}')

if __name__ == '__main__':
    #delete all files in the directories data/noiser/raw_exp, data/noiser/sim, data/noiser/sim_noisy, data/noiser/exp, data/noiser/train/exp, data/noiser/val/exp, data/noiser/train/noisy, data/noiser/val/noisy, data/noiser/train/clean, data/noiser/val/clean
    directories = [
        'data/noiser/raw_exp',
        'data/noiser/sim',
        'data/noiser/sim_noisy',
        'data/noiser/exp',
        'data/noiser/train/exp',
        'data/noiser/val/exp',
        'data/noiser/train/noisy',
        'data/noiser/val/noisy',
        'data/noiser/train/clean',
        'data/noiser/val/clean'
    ]

    delete_all_files_in_directories(directories)

    #raw file name, pixel_size, unique particle id, series_number, diameter, columns_visible
    data = [["1108",0.035895,1,1,2.5,1],
        ["1110_1",0.025382,1,2,2.5,1],
        ["1110_2",0.025382,1,3,2.5,1],
        ["1110_3",0.025382,1,4,2.5,1],
        ["1122",0.035895,2,5,2.2,0],
        ["1124_0001_1",0.025382,3,6,2.0,0],
        ["1124_0001_2",0.025382,4,7,2.3,0],
        ["1124_0001_3",0.025382,5,8,1.9,0],
        ["1124_1",0.050764,6,9,1.9,0],
        ["1124_2",0.050764,7,10,4.0,0],
        ["1127",0.025382,8,11,1.6,0],
        ["1128_1",0.025382,9,12,2.7,0],
        ["1128_2",0.025382,9,13,2.7,0],
        ["1128_3",0.025382,9,14,2.7,0],
        ["1128_4",0.025382,9,15,2.7,0],
        ["1128_5",0.025382,9,16,2.7,0],
        ["1137",0.035895,10,17,1.7,0],
        ["1138_1",0.025382,10,18,1.9,0],
        ["1138_2",0.025382,10,19,2.0,0],
        ["1139",0.025382,11,20,2.2,0],
        ["1139_0001",0.035895,11,21,2.2,0],
        ["1140",0.025382,12,22,1.5,0],
        ["1141",0.017948,12,23,1.5,0],
        ["1142_1",0.025382,13,24,2.0,1],
        ["1142_2",0.025382,13,25,2.0,1],
        ["1143",0.025382,13,26,2.0,1],
        ["1144",0.035895,13,27,2.0,1],
        ["1145",0.025382,14,28,2.8,1],
        ["1148",0.025382,15,29,2.6,0],
        ["1148_0001",0.035895,15,30,2.6,0],
        ["1149",0.035895,16,31,1.9,1],
        ["1152",0.025382,17,32,1.3,0],
        ["1155",0.017948,18,33,0.2,0],
        ["1156_1",0.025382,19,34,1.3,0],
        ["1156_2",0.025382,20,35,1.7,0],
        ["1157",0.017948,20,36,1.7,0],
        ["1157_0001_1",0.025382,20,37,1.7,0],
        ["1157_0001_2",0.025382,19,38,1.3,0],
        ["1157_0001_3",0.025382,19,39,1.3,0],
        ["1157_0001_4",0.025382,20,40,1.7,0],
        ["1300", 0.025382,21,41,1.6,0],
        ["1306", 0.035895,22,42,2.8,0],
        ["1307_1",0.025382,22,43,2.8,0],
        ["1307_2",0.025382,22,44,2.8,0],
        ["1307_3",0.025382,22,45,2.8,0],
        ["1307_4",0.025382,22,46,2.8,0],
        ["1307_5",0.025382,22,47,2.8,0],
        ["1307_6",0.025382,22,48,2.8,0],
        ["1307_7",0.025382,22,49,2.8,0],
        ["1308",0.025382,22,50,2.8,0],
        ["1311_1",0.025382,23,51,1.4,0],
        ["1311_2",0.025382,24,52,1.9,0],
        ["1312",0.025382,25,53,2.4,0],
        ["1312_0001", 0.017948, 26, 54, 1.8,0],
        ["1315",0.025382,27,55,2.0,0],
        ["1315_0001",0.035895,27,56,2.0,0],
        ["1316_1",0.025382,27,57,2.0,1],
        ["1316_2",0.025382,27,58,2.0,1],
        ["1317", 0.035895,27,59,2.0,0],
        ["1317_0001_1",0.017948, 27, 60, 2.0,1],
        ["1317_0001_2",0.017948, 27, 61, 2.0,0],
        ["1317_0001_3",0.017948, 27, 62, 2.0,0],
        ["1317_0001_4",0.017948, 27, 63, 2.0,0],
        ["1318", 0.025382, 27, 64, 2.0,0],
        ["1321_0001", 0.017948, 28, 65, 1.4,0],
        ["1325", 0.025382, 29, 66, 2.4,0],
        ["1325_0001", 0.035895, 29, 67, 2.4,0],
        ["1326", 0.025382, 29, 68, 2.4,0],
        ["1326_0001", 0.017948, 29, 69, 2.4,0],
        ["1328", 0.017948, 30, 70, 0.9,0],
        ["1330", 0.025382, 31, 71, 2.0,0],
        ["1331_1", 0.025382, 32, 72, 2.5,0],
        ["1331_2", 0.025382, 32, 73, 2.5,0],
        ["1338", 0.035895, 33, 74, 2.9,1],
        ["1339", 0.035895, 33, 75, 2.9,0],
        ["1339_0001", 0.025382, 33, 76, 2.9,0],
        ["1339_0002_1", 0.025382, 33, 77, 2.9,0],
        ["1339_0002_2", 0.025382, 33, 78, 2.9,0],
        ["1342", 0.025382, 34, 79, 1.6,0],
        ["1345", 0.025382, 35, 80, 2.4,0],
        ["1346",  0.025382, 35, 81, 2.4,0],
        ["1346_0001", 0.035895, 35, 82, 2.4,0],
        ["1349", 0.025382, 36, 83, 1.1,0],
        ["1350", 0.025382, 37, 84, 1.1,0],
        ["1350_0001", 0.017948, 37, 85, 1.1,0],
        ["1352", 0.035895, 38, 86, 2.8,0],
        ["1352_0001_1", 0.025382, 38, 87, 2.8,0],
        ["1352_0001_2", 0.025382, 38, 88, 2.8,0],
        ["1352_0001_3", 0.025382, 38, 89, 2.8,1],
        ["1353", 0.025382, 38, 90, 2.8,1],
        ["1358_1", 0.025382, 39, 91, 2.2,1],
        ["1358_2", 0.025382, 39, 92, 2.2,1],
        ["1358_3", 0.025382, 40, 93, 0.8,0],
        ["1359", 0.025382, 41, 94, 2.1,0],
        ["1406", 0.025382, 42, 95, 1.6,0],
        ["1406_0001", 0.017948, 42, 96, 1.6,0],
        ["1408", 0.025382, 43, 97, 0.5,0],
        ["1412", 0.035895, 44, 98, 2.2,1],
        ["1413", 0.025382, 44, 99, 2.2,1],
        ["1414", 0.025382, 44, 100, 2.2,1],
        ["1414_0001_1", 0.017948, 44, 101, 2.2,1],
        ["1414_0001_2", 0.017948, 44, 102, 2.2,1],
        ["1414_0001_3", 0.017948, 44, 103, 2.2,1],
        ["1414_0001_4", 0.017948, 44, 104, 2.2,1],
        ["1414_0001_5", 0.017948, 44, 105, 2.2,0],
        ["1417_1", 0.035895, 45, 106, 3.0,0],
        ["1417_2", 0.035895, 45, 107, 3.0,0],
        ["1418", 0.025382, 45, 108, 3.0,0],
        ["1419", 0.025382, 45, 109, 3.0,0],
        ["1428", 0.025382, 46, 110, 2.8,0],
        ["1429_0001", 0.035895, 46, 111, 2.8,0],
        ["1430", 0.035895, 47, 112, 1.8,0],
        ["1431", 0.025382, 47, 113, 1.8,0],
        ["1432", 0.017948, 47, 114, 1.8,0],
        ["1433", 0.025382, 48, 115, 1.3,1],
        ["1434", 0.017948, 48, 116, 1.3,1], #weird background value
        ["1437", 0.025382, 49, 117, 2.7,1],
        ["1437_0001", 0.035895, 49, 118, 2.7,1],
        ["1438_0001_1", 0.025382, 50, 119, 2.0,0],
        ["1438_0001_2", 0.025382, 50, 120, 2.0,0],
        ["1438_1", 0.035895, 51, 121, 2.2,0],
        ["1438_2", 0.035895, 52, 122, 1.9,0],
        ["1450", 0.035895, 53, 123, 1.9,0],
        ["1458_0001_1", 0.017948, 54, 124, 1.5,0],
        ["1458_0001_2", 0.017948, 55, 125, 0.9,1],
        ["1458_1", 0.025382, 54, 126, 1.5,0],
        ["1458_2", 0.025382, 55, 127, 0.9,0],
        ["1518", 0.025382, 56, 128, 1.9,0],
        ["1522", 0.035895, 57, 129, 2.2,1],
        ["1527", 0.035895, 57, 130, 2.2,1],
        ["1527_0001", 0.025382, 57, 131, 2.2,1],
        ["1529_0001_1",0.017948, 58, 132, 1.8,0],
        ["1529_0001_2", 0.017948, 59, 133, 1.4,0],
        ["1529_1", 0.025382, 60, 134, 1.7,0],
        ["1529_2", 0.025382, 61, 135, 1.4,0],
        ["1530", 0.017948, 62, 136, 1.0,1],
        ["1538", 0.025382, 63, 137, 1.8,0],
        ["1539", 0.017948, 63, 138, 1.8,0],
        ["1556", 0.025382, 64, 139, 2.0,1],
        ["1557", 0.035895, 64, 140, 2.0,0],
        ["1611", 0.035895, 65, 141, 2.0,1],
        ["1612", 0.035895, 65, 142, 2.0,1],
        ["1613", 0.025382, 65, 143, 2.0,1],
        ["1615", 0.025382, 65, 144, 2.0,1],
        ["1616", 0.025382, 65, 145, 2.0,1],
        ["1617_0001", 0.017948, 65, 146, 2.0,1],
        ["1624", 0.017948, 65, 147, 2.0,1],
        ["1624_0001", 0.025382, 65, 148, 2.0,1],
        ["1630", 0.025382, 66, 149, 1.8,1],
        ["1631", 0.035895, 66, 150, 1.8,0],
        ["1631_0001", 0.017948, 66, 151, 1.8,1],
        ["1633_1", 0.035895, 66, 152, 1.8,0],
        ["1633_2", 0.035895, 67, 153, 2.0,0],
        ["1634", 0.025382, 67, 154, 2.0,0],
        ["1639", 0.025382, 68, 155, 2.7,0],
        ["1640", 0.035895, 68, 156, 2.7,0],
        ["1641", 0.017948, 69, 157, 1.7,0],
        ["1642", 0.025382, 70, 158, 1.5,0],
        ["1647", 0.025382, 71, 159, 1.0,0],
        ["1655", 0.025382, 72, 160, 2.3,1], #weird background value
        ["1658", 0.025382, 72, 161, 2.3,1],
        ["1658_0001", 0.017948, 72, 162, 2.3,1],
        ["1701_0001", 0.035895, 73, 163, 3.1,0],
        ["1703", 0.025382, 74, 164, 1.6,1],
        ["1704", 0.035895, 75, 165, 4.1,0],
        ["1704_0001", 0.050764, 75, 166, 4.1,0],
        ["1706", 0.025382, 76, 167, 2.2,0],
        ["1707", 0.025382, 77, 168, 2.4,1],
        ["1708", 0.035895, 76, 169, 2.2,0],
        ["1708_c", 0.035895, 78, 170, 1.3,0],
        ["1709", 0.025382, 79, 171, 1.8,0],
        ["1709_0001", 0.017948, 79, 172, 1.8,0],
        ["1709_c", 0.035895, 80, 173, 2.2,0],
        ["1712_0002_c", 0.050764, 81, 174, 2.5,0],
        ["1712_c", 0.035895, 81, 175, 2.5,0],
        ["1713", 0.025382, 82, 176, 2.0,1],
        ["1714_0001_c", 0.025382, 83, 177, 2.9,0],
        ["1715_c", 0.035895, 83, 178, 2.9,0],
        ["1717_0001_c", 0.025382, 84, 179, 1.5,0],
        ["1717_c", 0.025382, 85, 180, 1.6,0],
        ["1721_c", 0.025382, 86, 181, 2.4,0],
        ["1722_0001_c", 0.050764, 87, 182, 3.9,0],
        ["1722_c", 0.035895, 87, 183, 3.9,0],
        ["1723_c", 0.025382, 88, 184, 1.1,0],
        ["1724", 0.025382, 89, 185, 1.8,1],
        ["1724_0001", 0.017948, 89, 186, 1.8,1],
        ["1724_c", 0.025382, 90, 187, 1.3,0],
        ["1725_0001_c", 0.035895, 91, 188, 3.4,0],
        ["1725_c", 0.050764, 91, 189, 3.4,0],
        ["1726_c", 0.035895, 92, 190, 1.9,0],
        ["1727_c", 0.025382, 92, 191, 1.9,0],
        ["1732_0001_c", 0.025382, 93, 192, 1.0,0],
        ["1732_c", 0.025382, 94, 193, 1.8,1],
        ["1733_c", 0.017948, 95, 194, 1.1,1],
        ["1734_0001_c", 0.025382, 96, 195, 1.7,0],
        ["1734_c", 0.017948, 96, 196, 1.7,0],
        ["1736", 0.035895, 97, 197, 2.0,1],
        ["1736_c", 0.025382, 98, 198, 1.7,0],
        ["1737", 0.025382, 97, 199, 2.0,1],
        ["1737_0001_1", 0.025382, 97, 200, 2.0,1], #Good for atom counting
        ["1737_0001_2", 0.025382, 97, 201, 2.0,1], #Good for atom counting
        ["1738", 0.017948, 97, 202, 2.0,1], #Good for atom counting
        ["1739", 0.025382, 99, 203, 2.3,0],
        ["1739_1_c", 0.035895, 100, 204, 2.7,0],
        ["1739_2_c", 0.035895, 101, 205, 1.6,0],
        ["1740_0001_c", 0.025382, 102, 206, 2.0,1],
        ["1740_c", 0.025382, 101, 207, 1.6,1],
        ["1743_c", 0.025382, 103, 208, 2.1,1],
        ["1744_c", 0.025382, 104, 209, 2.1,1],
        ["1748_0001_c", 0.025382, 104, 210, 2.1,1],
        ["1748_c", 0.025382, 104, 211, 2.1,1],
        ["1749_0001_1_c", 0.017948, 105, 212, 1.0,1],
        ["1749_0001_2_c", 0.017948, 105, 213, 1.0,1],
        ["1749_c", 0.025382, 105, 214, 1.0,1],
        ["1753_c", 0.035895, 106, 215, 1.8,1],
        ["1754_0001", 0.025382, 107, 216, 2.3,1],
        ["1754_c", 0.025382, 106, 217, 1.8,1],
        ["1756_c", 0.035895, 108, 218, 2.8,0],
        ["1757_c", 0.050764, 109, 219, 3.2,0],
        ["1758_c", 0.035895, 109, 220, 3.2,0],
        ["1800_c", 0.035895, 110, 221, 2.3,0],
        ["1801_c", 0.025382, 110, 222, 2.3,0],
        ["1802_c", 0.025382, 111, 223, 1.9,0],
        ["1803_0001_c", 0.025382, 111, 224, 1.9,0],
        ["1803_c", 0.017948, 111, 225, 1.9,0],
        ["1805_0001_c", 0.035895, 112, 226, 2.7,1],
        ["1805_c", 0.025382, 112, 227, 2.7,1],
        ["1806_c", 0.025382, 113, 228, 2.2,0],
        ["1807", 0.025382, 114, 229, 0.7,0],
        ["1807_c", 0.025382, 115, 230, 2.0,0],
        ["1818", 0.025382, 116, 231, 2.8,0],
        ["1822", 0.025382, 117, 232, 1.7,0],
        ["1832", 0.025382, 118, 233, 2.1,0],
        ["1833", 0.017948, 118, 234, 2.1,0],
        ["3_1245", 0.050764, 119, 235, 1.7,0],
        ["3_1246", 0.025382, 119, 236, 1.7,0],
        ["3_1248", 0.035895, 120, 237, 1.9,0],
        ["3_1304_0001", 0.025382, 121, 238, 0.7,1],
        ["3_1305", 0.025382, 122, 239, 0.7,1],
        ["3_1306", 0.035895, 123, 240, 1.7,0],
        ["3_1308", 0.025382, 123, 241, 1.7,0],
        ["3_1308_2", 0.025382, 124, 242, 1.2,0],
        ["3_1310", 0.025382, 125, 243, 0.4,0],
        ["3_1311", 0.025382, 126, 244, 2.1,0],
        ["3_1315", 0.025382, 127, 245, 2.4,1],
        ["3_1316", 0.025382, 127, 246, 2.4,1],
        ["3_1316_0001", 0.025382, 127, 247, 2.4,1],
        ["3_1317", 0.025382, 128, 248, 1.0,1],
        ["3_1318", 0.035895, 127, 249, 2.4,1],
        ["3_1319_0001_1", 0.035895, 129, 250, 2.2,0],
        ["3_1319_0001_2", 0.035895, 130, 251, 0.7,0],
        ["3_1319", 0.025382, 131, 252, 0.7,1],
        ["3_1320_0001", 0.025382, 132, 253, 1.7,0],
        ["3_1320", 0.025382, 129, 254, 2.2,0],
        ["3_1323", 0.025382, 133, 255, 0,1],
        ["3_1324", 0.025382, 134, 256, 1.8,0],
        ["3_1325_0001", 0.035895, 135, 257, 4.3,0],
        ["3_1325_0002", 0.025382, 136, 258, 1.1,0],
        ["3_1325", 0.035895, 134, 259, 1.8,0],
        ["3_1326", 0.025382, 137, 260, 1.9,0],
        ["3_1327_0001", 0.017948, 138, 261, 1.6,0],
        ["3_1327", 0.025382, 138, 262, 1.6,0],
        ["3_1328", 0.025382, 139, 263, 2.5,1],
        ["3_1329_1", 0.017948, 140, 264, 1.3,1],
        ["3_1329_2", 0.017948, 141, 265, 1.1,1],
        ["3_1330_0001", 0.025382, 142, 266, 1.0,1],    
        ["3_1330_0002", 0.017948, 142, 267, 1.0,1],
        ["3_1330_1", 0.025382, 140, 268, 1.3,1],
        ["3_1330_2", 0.025382, 141, 269, 1.1,1],
        ["3_1331_1", 0.025382, 143, 270, 1.3,0],
        ["3_1331_2", 0.025382, 144, 271, 1.0,0],
        ["3_1332", 0.017948, 144, 272, 1.0,0],
        ["3_1333_0001", 0.017948, 145, 273, 1.0,0],
        ["3_1333", 0.025382, 145, 274, 1.0,0],
        ["3_1337", 0.035895, 146, 275, 2.2,1],
        ["3_1338_0001", 0.025382, 146, 276, 2.2,1],
        ["3_1338", 0.035895, 146, 277, 2.2,1],
        ["3_1340_0001", 0.025382, 146, 278, 2.2,1],
        ["3_1340", 0.035895, 146, 279, 2.2,1],
        ["3_1341", 0.050764, 146, 280, 2.2,1],
        ["3_1342_0001", 0.017948, 147, 281, 1.4,1],
        ["3_1342", 0.025382, 147, 282, 1.4,1],
        ["3_1343", 0.025382, 147, 283, 1.4,1],
        ["3_1345", 0.025382, 146, 284, 2.2,1],
        ["3_1347_0001", 0.025382, 148, 285, 0.7,0],
        ["3_1347", 0.025382, 149, 286, 1.1,1],
        ["3_1350_0001", 0.025382, 150, 287, 2.3,0],
        ["3_1350", 0.025382, 151, 288, 1.7,1],
        ["3_1351_1", 0.025382, 152, 289, 1.9,1],
        ["3_1351_2", 0.025382, 153, 290, 1.4,0],
        ["3_1352", 0.017948, 154, 291, 0.9,0],
        ["3_1353", 0.035895, 155, 292, 2.0,0],
        ["3_1354_0001", 0.025382, 156, 293, 1.2,0],
        ["3_1354_1", 0.025382, 157, 294, 1.8,0],
        ["3_1354_2", 0.025382, 158, 295, 2.1,0],
        ["3_1356", 0.025382, 159, 296, 2.1,0],
        ["3_1357_0001", 0.025382, 160, 297, 1.7,0],
        ["3_1357", 0.025382, 160, 298, 1.7,0],
        ["3_1358", 0.025382, 161, 299, 2.3,0],
        ["3_1359", 0.035895, 161, 300, 2.3,0],
        ["3_1400", 0.035895, 162, 301, 2.1,0],
        ["3_1401", 0.025382, 162, 302, 2.1,0],
        ["3_1402", 0.025382, 163, 303, 0.5,0],
        ["3_1403", 0.035895, 164, 304, 2.9,0],
        ["3_1405", 0.035895, 165, 305, 2.9,0],
        ["3_1413_1", 0.035895, 166, 306, 2.9,0],
        ["3_1413_2", 0.035895, 167, 307, 3.0,0],
        ["3_1414_0001", 0.025382, 168, 308, 2.0,0],
        ["3_1414", 0.025382, 167, 309, 3.0,0],
        ["3_1416_0001", 0.025382, 169, 310, 2.7,1], #Amazing
        ["3_1416", 0.035895, 169, 311, 2.7,1],
        ["3_1418_0001", 0.050764, 169, 312, 2.7,1],
        ["3_1418", 0.050764, 169, 313, 2.7,1],
        ["3_1419", 0.017948, 170, 314, 0.5,1],
        ["3_1420_0001", 0.035895, 171, 315, 1.9,1],
        ["3_1420", 0.025382, 171, 316, 1.9,1],
        ["3_1421", 0.025382, 172, 317, 1.9,0],
        ["3_1422", 0.035895, 172, 318, 1.9,0],
        ["3_1423", 0.025382, 173, 319, 2.4,0],
        ["3_1424", 0.025382, 173, 320, 2.4,0],
        ["3_1427", 0.025382, 174, 321, 1.3,0],   
        ["3_1428", 0.025382, 175, 322, 2.2,0],
        ["3_1430_0001", 0.025382, 176, 323, 2.2,0],
        ["3_1430_1", 0.025382, 177, 324, 1.2,0],
        ["3_1430_2", 0.025382, 178, 325, 1.9,0],
        ["3_1431", 0.025382, 176, 326, 2.2,0],
        ["3_1433", 0.025382, 179, 327, 2.0,1],
        ["3_1434", 0.035895, 180, 328, 1.8,0],
        ["3_1435", 0.035895, 181, 329, 1.8,1],
        ["3_1437_1", 0.050764, 182, 330, 3.4,0],
        ["3_1437_2", 0.050764, 183, 331, 1.8,0],
        ["3_1438_1", 0.035895, 182, 332, 3.4,0],
        ["3_1438_2", 0.035895, 183, 333, 1.8,0],
        ["3_1439", 0.025382, 184, 334, 1.6,0],
        ["3_1441",0.035895, 185, 335, 2.1,1],
        ["3_1442", 0.025382, 185, 336, 2.1,1],
        ["3_1443_0001", 0.025382, 186, 337, 2.3,1],
        ["3_1443", 0.035895, 186, 338, 2.3,1],
        ["3_1444", 0.025382, 187, 339, 1.5,0],
        ["3_1446", 0.025382, 188, 340, 2.0,0],
        ["3_1448_1", 0.035895, 189, 341, 3.1,0],
        ["3_1448_2", 0.035895, 190, 342, 1.9,0],
        ["3_1449", 0.035895, 191, 343, 2.0,0],
        ["3_1451", 0.017948, 192, 344, 0.7,1],
        ["3_1452", 0.025382, 193, 345, 2.5,0],
        ["3_1453", 0.025382, 194, 346, 2.3,0],
        ["3_1454_1", 0.025382, 195, 347, 1.9,0],
        ["3_1454_2", 0.025382, 196, 348, 2.5,0],
        ["3_1457", 0.025382, 197, 349, 1.8,0],
        ["3_1458", 0.025382, 198, 350, 1.0,0],
        ["3_1459_0001", 0.025382, 199, 351, 2.1,1],
        ["3_1459", 0.035895, 199, 352, 2.1,0],
        ["3_1501", 0.025382, 200, 353, 1.8,0],
        ["3_1502", 0.017948, 200, 354, 1.8,0],
        ["3_1509", 0.025382, 201, 355, 1.9,1],
        ["3_1510", 0.035895, 201, 356, 1.9,1],
        ["3_1511", 0.025382, 202, 357, 0.7,1],
        ["3_1512", 0.025382, 203, 358, 1.7,1],
        ["3_1519", 0.025382, 204, 359, 1.2,1],
        ["3_1521", 0.017948, 205, 360, 2.1,1],
    ]
    df = pd.DataFrame(data,columns=["raw_file","pixel_size","unique_particle_id","series_number","diameter","columns_visible"])

    #how many series have columns_visible = 1
    print(df.columns_visible.value_counts())

    # Find all tif stacks in experimental data/new_set
    experimental_path = 'data/experimental/new_set'
    save_path = 'data/noiser/raw_exp'
    
    # use the file names in exp_info and add the same file from experimental_path to list_of_files
    list_of_files = [f"{row.raw_file}.tif" for _, row in df.iterrows()]
 
    # unpack the tif stacks to save_path and rename them to j_i.tif where j is the global frame index and i is the series number
    j = 1
    i = 1
    for file in list_of_files:
        tif_stack = tiff.imread(os.path.join(experimental_path, file))
        print(file, tif_stack.shape)
        for k in range(tif_stack.shape[0]):
            tiff.imsave(os.path.join(save_path, f"{j}_{i}.tif"), tif_stack[k])
            j += 1
            print(j)
        i += 1

    files = os.listdir(save_path)
    df["file_list"] = None
    for idx, row in df.iterrows():
        file_names = [file for file in files if file.split("_")[1].split(".")[0] == str(row.series_number)]
        #print(file_names)
        #sort based on file.split("_")[0], lowest at idx 0
        file_names.sort(key=lambda x: int(x.split("_")[0]))
        file_list = [file.split(".")[0] + ".npz" for file in file_names]

        # Update the dataframe
        df.at[idx, "file_list"] = file_list
        
    with open("data/exp_info.pkl", "wb") as f:
        pkl.dump(df,f)


    """
    Add Gaussian and Poisson noise to the simulated data and populate the folders 
    data/noiser
    │
    ├── exp
    │
    ├── sim
    │
    ├── sim_noisy
    │
    ├── train
    │   ├── clean
    │   ├── noisy
    │   └── exp
    │
    └── val
        ├── clean
        ├── noisy
        └── exp
    """

    # Read all mrc from noiser/raw_sim and save as npz in sim
    raw_sim_dir = "data/noiser/raw_sim/"
    raw_exp_dir = "data/noiser/raw_exp/"
    exp_dir = "data/noiser/exp/"
    sim_dir = "data/noiser/sim/"
    noisy_dir = "data/noiser/sim_noisy/"

    if len(os.listdir(raw_sim_dir)) == len(os.listdir(sim_dir)) and len(os.listdir(sim_dir)) != 0:
        print("Raw files already converted")
    else:
        convert_raw_to_npz(raw_sim_dir,sim_dir, apply_probe=True)

    if len(os.listdir(raw_exp_dir)) == len(os.listdir(exp_dir)) and len(os.listdir(exp_dir)) != 0:
        print("Raw files already converted")
    else:
        convert_exp_to_npz(raw_exp_dir,exp_dir)

    # Add noise to the simulated images and save in sim_noisy

    if len(os.listdir(sim_dir)) == len(os.listdir(noisy_dir)) and len(os.listdir(noisy_dir)) != 0:
        print("Noisy data already created")
    else:
        add_noise(sim_dir, noisy_dir)

    # Normalize sets by min max normalization within set
    normalize(sim_dir)
    normalize(noisy_dir)
    normalize_exp(exp_dir)    
    print("Normalization complete")
    #Split into train and val
    assert len(os.listdir(sim_dir)) == len(os.listdir(noisy_dir))
    
    train_files, val_files = split_and_save(0.8, sim_dir, "clean",return_split=True)
    split_and_save(0.8, noisy_dir, "noisy", split = [train_files, val_files])

    split_and_save_exp(exp_dir, sim_dir, split=0.8)

    print("Split complete")

    with open("data/exp_info.pkl", "rb") as f:
        exp_info = pkl.load(f)
    print(exp_info.split.value_counts())


#------------Prep data^^^^^^-----------#