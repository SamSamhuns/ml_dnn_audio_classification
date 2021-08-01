import os
import glob
import argparse
import soundfile

from tqdm import tqdm
import os.path as osp
from util import _fix_path_for_globbing

# #################### Raw Data Organization ########################
#   raw_data
#          |_ dataset
#                   |_ class_1
#                             |_ img1
#                             |_ img2
#                             |_ ....
#                   |_ class_2
#                             |_ img1
#                             |_ img2
#                             |_ ....
#                   ...
####################################################################

# #################### Data configurations here #####################
VALID_FILE_EXTS = {'wav', 'mp3'}
####################################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd',
                        '--raw_data_path',
                        type=str,
                        required=True,
                        help="""Raw dataset path with
                        class imgs inside folders""")
    parser.add_argument('-td',
                        '--target_data_path',
                        type=str,
                        required=True,
                        help="""Target dataset path where
                        imgs will be saved in sub folders
                        repr classes with number matching target_number""")
    parser.add_argument('-b',
                        '--target_bit_depth',
                        type=str,
                        required=False,
                        default="PCM_16",
                        help="""Default PCM_16. Target bit depth for audio samples""")
    args = parser.parse_args()
    convert_to_bitdepth(args.raw_data_path,
                        args.target_data_path,
                        args.target_bit_depth)


def save_audio_with_bit_depth(file_path: str, target_path: str, bdepth: str = "PCM_16") -> None:
    """Save aduio file from file_path to target_path in bdepth bitdepths
    """
    data, sr = soundfile.read(file_path)
    soundfile.write(target_path, data, sr, subtype=bdepth)


def convert_to_bitdepth(RAW_IMG_DIR, DUPLICATED_IMG_DIR, TARGET_BIT_DEPTH) -> None:
    target_dir = DUPLICATED_IMG_DIR
    os.makedirs(target_dir, exist_ok=True)

    dir_list = glob.glob(_fix_path_for_globbing(RAW_IMG_DIR))

    # for each class in raw data
    for i in tqdm(range(len(dir_list))):
        dir = dir_list[i]                # get path to class dir
        class_name = dir.split("/")[-1]  # get class name
        f_list = [file for file in sorted(glob.glob(dir + "/*"))
                  if file.split(".")[-1] in VALID_FILE_EXTS]

        class_target_dir = osp.join(target_dir, class_name)

        # skip copying if dir already exists and has required num of files
        if osp.exists(class_target_dir):
            tf_list = glob.glob(class_target_dir + "/*")
            if len(tf_list) >= len(f_list):
                continue
        os.makedirs(class_target_dir, exist_ok=True)

        for file_path in tqdm(f_list):
            bname = osp.basename(file_path)
            target_path = osp.join(class_target_dir, bname)
            save_audio_with_bit_depth(file_path, target_path, TARGET_BIT_DEPTH)


if __name__ == "__main__":
    main()
