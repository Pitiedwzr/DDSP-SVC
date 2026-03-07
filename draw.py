import argparse
import numpy as np
import tqdm
import os
import shutil

import soundfile as sf

WAV_MIN_LENGTH = 2    # wav文件的最短时长 / The minimum duration of wav files
SAMPLE_MIN = 1    # 抽取的文件数量下限 / The lower limit of the number of files to be extracted
SAMPLE_MAX = 1    # 抽取的文件数量上限 / The upper limit of the number of files to be extracted


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    root_dir = os.path.abspath('.')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--train",
        type=str,
        default=root_dir + "/data/train/audio", # 固定源目录为根目录下/data/train/audio目录
        help="directory where contains train dataset"
    )
    parser.add_argument(
        "-v",
        "--val",
        type=str,
        default=root_dir + "/data/val/audio",
        help="directory where contains validate dataset"
    )
    parser.add_argument(
        "-r",
        "--sample_rate",
        type=float,
        default=1,
        help="The percentage of files to be extracted"  # 抽取文件数量的百分比
    )
    parser.add_argument(
        "-e",
        "--extensions",
        type=str,
        required=False,
        nargs="*",
        default=["wav", "flac"],
        help="list of using file extensions, e.g.) -f wav flac ..."
    )
    return parser.parse_args(args=args, namespace=namespace)


# 定义一个函数，用于检查wav文件的时长是否大于最短时长
def check_duration(wav_file):
    # 打开wav文件
    f = sf.SoundFile(wav_file)
    # 获取帧数和帧率
    frames = f.frames
    rate = f.samplerate
    # 计算时长（秒）
    duration = frames / float(rate)
    # 关闭文件
    f.close()
    # 返回时长是否大于最短时长的布尔值
    return duration > WAV_MIN_LENGTH

# 定义一个函数，用于从给定的目录中随机抽取一定比例的wav文件，并剪切到另一个目录中，保留数据结构
def split_data(src_dir, dst_dir, ratio, extensions):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    subdirs, files = [], []
    for item in os.listdir(src_dir):
        item_path = os.path.join(src_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
        elif os.path.isfile(item_path) or os.path.islink(item_path): # 增加对 link 的识别
            if any([item.endswith(f".{ext}") for ext in extensions]):
                files.append(item)

    if len(files) > 0:
        # 计算数量
        num_files = int(len(files) * ratio)
        num_files = max(SAMPLE_MIN, min(SAMPLE_MAX, num_files))

        np.random.shuffle(files)
        selected_files = files[:num_files]
        
        pbar = tqdm.tqdm(total=len(selected_files), desc=f"Processing {os.path.basename(src_dir)}")

        for file in selected_files:
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)

            # 1. 检查时长
            if not check_duration(src_file):
                print(f"Skipped {src_file} (too short)")
                continue
            
            # 2. 【核心修改】复制真实内容而非移动软链接
            # shutil.copy 会跟随软链接，将原始只读目录里的音频内容复制到 temp/val 中
            shutil.copy(src_file, dst_file)
            
            # 3. 【核心修改】从训练集中删除这个软链接（防止训练集/验证集数据重合）
            # 因为 src_dir 在 /kaggle/temp 下，所以我们有权限删除这个 link
            os.remove(src_file)
            
            pbar.update(1)
        pbar.close()

    # 递归处理子目录
    for subdir in subdirs:
        split_data(os.path.join(src_dir, subdir), os.path.join(dst_dir, subdir), ratio, extensions)

# 定义主函数，用于获取用户输入并调用上述函数

def main(cmd):
    dst_dir = cmd.val
    # 抽取比例，默认为1
    ratio = cmd.sample_rate / 100

    src_dir = cmd.train
    
    extensions = cmd.extensions

    # 调用split_data函数，对源目录中的wav文件进行抽取，并剪切到目标目录中，保留数据结构
    split_data(src_dir, dst_dir, ratio, extensions)

# 如果本模块是主模块，则执行主函数
if __name__ == "__main__":
    # parse commands
    cmd = parse_args()
    
    main(cmd)
