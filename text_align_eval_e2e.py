# load packages
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import warnings
import soundfile as sf
import argparse
warnings.simplefilter('ignore')
from tqdm import tqdm


from Utils.ASR.models import ASRCNN

from models import *
from losses import *
from utils import *

from scipy.stats import entropy
import random

device = 'cuda'

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kl_divergence(arr1, arr2):
    """
    计算两个一维数组的 KL 散度
    :param arr1: 第一个一维数组
    :param arr2: 第二个一维数组
    :return: KL 散度
    """
    # 为了避免 KL 散度计算时出现除以零的错误，添加一个小的平滑项
    epsilon = 0.01
    arr1 = arr1 + epsilon
    arr2 = arr2 + epsilon
    arr1 = arr1 / np.sum(arr1)
    arr2 = arr2 / np.sum(arr2)
    return entropy(arr1, arr2)

def set_min_to_value(arr, specified_value):
    min_value = np.min(arr)
    arr[arr == min_value] = specified_value
    return arr

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                continue
                # print(char)
        return indexes

textclenaer = TextCleaner()


import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

def read_lab_file(file_path):
    """
    读取 .lab 文件并返回内容
    :param file_path: .lab 文件的路径
    :return: 文件内容的列表
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 去掉每行末尾的换行符
    lines = [line.strip() for line in lines]
    return lines


# load pretrained ASR model
ASR_config = 'Utils/ASR/config.yml'
ASR_path = 'Utils/ASR/epoch_00080.pth'
text_aligner = load_ASR_models(ASR_path, ASR_config)

text_aligner.to(device)

n_down = text_aligner.n_down

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def find_file_in_directory(directory, filename):
    for root, _, files in os.walk(directory):
        for file in files:
            if file == filename:
                return os.path.join(root, file)
    print('{} not found'.format(filename))
    return None  # 如果找不到文件，返回None

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def load_tensor(wav_path):
    wave, sr = sf.read(wav_path)
    if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
    if sr != 24000:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
        # print(wave_path, sr)
        
    wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
    mel_tensor = preprocess(wave).squeeze()

    return mel_tensor

def get_duration(wav_path, phoneme_idx):

    set_all_random_seed(1234)

    mel_tensor = load_tensor(wav_path)
    length_mel = mel_tensor.shape[1]
    mel_tensor = mel_tensor[:, :(length_mel - length_mel % 2)]

    mel_tensor = mel_tensor.unsqueeze(0)
    text_tensor = torch.LongTensor(phoneme_idx)
    text_tensor = text_tensor.unsqueeze(0)

    input_length = torch.Tensor([text_tensor.shape[-1]])
    text_mask = length_to_mask(input_length).to(device)
    mel_length = torch.Tensor([mel_tensor.shape[-1]])
    mel_mask = length_to_mask(mel_length).to(device)
    
    mask = length_to_mask(mel_length // (2 ** n_down)).to(device)

    try:
        _, _, s2s_attn = text_aligner(mel_tensor.to(device), mask.to(device), text_tensor.to(device))
        s2s_attn = s2s_attn.transpose(-1, -2)
        s2s_attn = s2s_attn[..., 1:]
        s2s_attn = s2s_attn.transpose(-1, -2)
    except:
        # print('{} error'.format(wav_path))
        return None

    mask_ST = mask_from_lens(s2s_attn, input_length, mel_length // (2 ** n_down))
    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

    duration_pred = s2s_attn_mono.sum(axis=-1).detach()

    return duration_pred.squeeze().cpu().numpy()




def eval(pred_path, target_path):

    results = []
 
    output_file_list = os.listdir(pred_path)
    for output_file in tqdm(output_file_list):
        try:
            if output_file[0:9] == 'wav_pred_':
                basename = output_file[9:]
            else:
                basename = output_file
            output_wav_path = os.path.join(pred_path, output_file)
            gt_wav_path = find_file_in_directory(target_path, basename)
            text_path = find_file_in_directory(target_path, basename.replace('.wav', '.lab'))

            raw_text = read_lab_file(text_path)[0]

            phoneme = global_phonemizer.phonemize([raw_text])[0]
            phoneme_idx = textclenaer(phoneme)
            phoneme_idx.insert(0, 0)
            phoneme_idx.append(0)

            duration_pred = get_duration(output_wav_path, phoneme_idx)
            duration_gt = get_duration(gt_wav_path, phoneme_idx)

            duration_pred = set_min_to_value(duration_pred, 1)
            duration_gt = set_min_to_value(duration_gt, 1) 
            
            results.append(kl_divergence(duration_gt, duration_pred))

        except Exception:
            print("{} not normal".format(output_file))
            continue
    
    results = np.array(results)
    print("The KL divergence between {} and gt duration and is {:.4f}, total test num: {}".format(pred_path, results.mean(), len(results)))

        

parser = argparse.ArgumentParser()
    # parser.add_argument("--restore_step", type=int, default=0)
parser.add_argument(
    "-p",
    "--pred_path",
    type=str,
    help="path to pred wav files",
)
parser.add_argument(
    "-t", 
    "--target_path", 
    type=str,
    required=True, 
    help="path to the target wav files",
)
args = parser.parse_args()

eval(args.pred_path, args.target_path)

















