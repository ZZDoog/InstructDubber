import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)
import json

import numpy as np
np.random.seed(0)

# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
import soundfile as sf

from models import *
from utils import *
from infer_utils import get_visual_feature_from_Chem, get_visual_feature_from_GRID, get_visual_feature_from_V2C
from text_utils import TextCleaner
textclenaer = TextCleaner()
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--restore_step", type=int, default=0)
parser.add_argument(
    "-n",
    "--exp_name",
    type=str,
    # required=True,
    default='GRID_Instruct',
)
parser.add_argument( 
    "--epoch", 
    type=int,
    # required=True, 
    default='16',
)
parser.add_argument( 
    "--setting", 
    type=str,
    required=True, 
)
parser.add_argument( 
    "--alpha", 
    type=float,
    # required=True, 
    default=0.0,
)
parser.add_argument( 
    "--beta", 
    type=float,
    # required=True, 
    default=0.3
)

args = parser.parse_args()

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path, return_mel_len=False):
    wave, sr = librosa.load(path, sr=24000)
    mel_len = preprocess(wave).to(device).shape[-1]
    if return_mel_len:
        return _ , mel_len
    # Trying remove the audio trim
    # wave, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
    mel_tensor = preprocess(wave).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1), mel_len

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# load phonemizer
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

config = yaml.safe_load(open("output/{}/config_grid.yml".format(args.exp_name)))
data_params = config.get('data_params', None)

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])
# if args.instruct_emotion:
model = build_model_Instruct(model_params, config, text_aligner, pitch_extractor, plbert)
# else:
#     model = build_model_GRID(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

params_whole = torch.load("output/{}/ckpt/epoch_2nd_{}.pth".format(args.exp_name, args.epoch), map_location='cpu')
params = params_whole['net']

for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)


def inference(text, ref_s, alpha = 0.0, beta = 0.3, diffusion_steps=5, embedding_scale=1, 
              mel_len=None, emotion_feature=None, lip_feature=None, atm_feature=None):

    tokens = torch.LongTensor(text).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        prosody_phoneme_feature = model.bert_encoder(bert_dur).transpose(-1, -2) 

        # Only for Ablation Study
        # prosody_phoneme_feature = t_en

        emotion_feature = emotion_feature.unsqueeze(0)
        lip_feature = lip_feature.unsqueeze(0)
        # atm_feature = atm_feature.unsqueeze(0)
        # prosody_phoneme_feature_emotion = model.prosody_fusion(prosody_phoneme_feature, text_mask, length_to_mask(torch.LongTensor([emotion_feature.shape[1]])).to(device), emotion_feature, atm_feature) + prosody_phoneme_feature
        prosody_phoneme_feature_emotion = model.prosody_fusion(prosody_phoneme_feature, text_mask, length_to_mask(torch.LongTensor([emotion_feature.shape[1]])).to(device), emotion_feature) + prosody_phoneme_feature

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                                        #   embedding=bert_dur,
                                          embedding=prosody_phoneme_feature_emotion.transpose(-1, -2),
                                          embedding_scale=embedding_scale,
                                        #   features=ref_s, # reference from the same speaker as the embedding
                                          num_steps=diffusion_steps).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        duration = model.duration_predictor(prosody_phoneme_feature,
                                            lip_feature,
                                            input_lengths,
                                            text_mask,
                                            length_to_mask(torch.LongTensor([lip_feature.shape[1]])).to(device) # Visual Mask
                                            )

        duration = torch.sigmoid(duration).sum(axis=-1)
        ori_duration_pred = torch.round(duration.squeeze())
        # pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        duration_sum = mel_len
        duration_logits = duration / duration.sum()
        duration = (duration_logits * duration_sum) / 2
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        # Eliminate potential noise at the end of the audio during generation.
        # if not text[-1].isalnum():
        pred_dur[0] += (pred_dur[-1]-1)
        pred_dur[-1] = 1

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        d = model.predictor.text_encoder(prosody_phoneme_feature_emotion, 
                                         s, input_lengths, text_mask)
        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, 
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))
    
        
    return out.squeeze().cpu().numpy()[..., :-50], ori_duration_pred # weird pulse at the end of the model, need to be fixed later


alpha = args.alpha
beta = args.beta
INSTRUCT_EMOTION = config.get('instruct_emotion', False)
INSTRUCT_DURATION = config.get('instruct_duration', False)


if INSTRUCT_DURATION:
    instruct_duration_mode = config.get('instruct_duration_path', 'Instruct_Lip_Raw')
    print('Instruct duration Mode: {}'.format(instruct_duration_mode))

# print("Generation Hyper-parameter: Aplha: {}, Beta: {}".format(alpha, beta))

duration_pred = []

save_path = 'output/{}/result_epoch_{}_grid2{}_align'.format(args.exp_name, args.epoch, args.setting)
os.makedirs(save_path, exist_ok=True)
print('Audio save path: {}'.format(save_path))

if args.setting == 'V2C':

    phoneme_data = 'dataset/V2C/V2C_duration_gt_val.json'
    with open(phoneme_data, 'r') as f1:
        data1 = [json.loads(line) for line in f1]
    path_to_phoneme_idx = {item['path']: item['phoneme_idx'] for item in data1}


    val_path = "/data1/home/zhangzhedong/preprocessed_data/Denoise_version2_all_feature_V2C/val.txt"
    wav_path = "/home/zhangzhedong/dvector/wav_22050_chenqi_clean_Denoise_version2_all"

    with open(val_path, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)

    noise = torch.randn(1,1,256).to(device)
    for i in tqdm(range(len(name))):

        phoneme_idx = path_to_phoneme_idx['{}/{}.wav'.format(speaker[i], name[i])]

        lip_feature, emotion_feature = get_visual_feature_from_V2C(speaker[i], name[i], 
                                                                   instruct_emotion = INSTRUCT_EMOTION,
                                                                   instruct_emotion_mode = 'GRID2V2C',
                                                                   instruct_duration = INSTRUCT_DURATION,
                                                                   instruct_duration_mode = instruct_duration_mode)
        gt_wav_path = "{}/{}/{}.wav".format(wav_path, speaker[i], name[i])

        ref_s, mel_len = compute_style(gt_wav_path)
        wav, dur_pred = inference(phoneme_idx, ref_s, alpha=alpha, beta=beta, embedding_scale=1, 
                        mel_len=mel_len, emotion_feature=emotion_feature, lip_feature=lip_feature)
        sf.write('{}/wav_pred_{}.wav'.format(save_path, name[i]), wav, 24000)
        duration_pred.append({"path": "{}/{}.wav".format(speaker[i], name[i]), "prediction":dur_pred.tolist()})

if args.setting == 'Chem':

    phoneme_data = 'dataset/Chem/Chem_duration_gt_val.json'
    with open(phoneme_data, 'r') as f1:
        data1 = [json.loads(line) for line in f1]
    path_to_phoneme_idx = {item['path']: item['phoneme_idx'] for item in data1}

    # save_path = 'LoRA_output/result_grid2{}_align'.format(args.setting)
    # os.makedirs(save_path, exist_ok=True)

    val_path = "/data1/home/zhangzhedong/preprocessed_data/Chem_Feature2025/chem_newphoneme_Test.txt"
    wav_path = "/data1/home/zhangzhedong/preprocessed_data/Chem_Feature2025/Chem_wav16_new"

    with open(val_path, "r", encoding="utf-8") as f:
            name = []
            path = []
            text = []
            raw_text = []
            speaker = []
            for line in f.readlines():
                n, p, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(n[:-4])
                path.append(p)
                text.append(t)
                raw_text.append(r)

    noise = torch.randn(1,1,256).to(device)
    for i in tqdm(range(len(name))):

        phoneme_idx = path_to_phoneme_idx['{}/00/{}.wav'.format(speaker[i], name[i])]

        lip_feature, emotion_feature = get_visual_feature_from_Chem(speaker[i], name[i], 
                                                                    instruct_emotion = INSTRUCT_EMOTION,
                                                                    instruct_emotion_mode = 'GRID2Chem',
                                                                    instruct_duration = INSTRUCT_DURATION,
                                                                    instruct_duration_mode = instruct_duration_mode)
        gt_wav_path = "{}/{}/00/{}.wav".format(wav_path, speaker[i], name[i])

        ref_s, mel_len = compute_style(gt_wav_path)
        wav, dur_pred = inference(phoneme_idx, ref_s, alpha=alpha, beta=beta, embedding_scale=1, 
                        mel_len=mel_len, emotion_feature=emotion_feature, lip_feature=lip_feature)
        sf.write('{}/wav_pred_{}.wav'.format(save_path, name[i]), wav, 24000)
        duration_pred.append({"path": "{}/00/{}.wav".format(speaker[i], name[i]), "prediction":dur_pred.tolist()})
            

if args.setting == 'GRID':

    phoneme_data = 'dataset/GRID/GRID_duration_gt_val.json'
    with open(phoneme_data, 'r') as f1:
        data1 = [json.loads(line) for line in f1]
    path_to_phoneme_idx = {item['path']: item['phoneme_idx'] for item in data1}

    val_path = "/data1/home/zhangzhedong/preprocessed_data/Grid_Wav_22050_Abs_Feature/val.txt"
    wav_path = "/data1/home/zhangzhedong/preprocessed_data/Grid_Wav_22050_Abs"

    with open(val_path, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)

    noise = torch.randn(1,1,256).to(device)
    for i in tqdm(range(len(name))):

        phoneme_idx = path_to_phoneme_idx['{}/{}.wav'.format(speaker[i], name[i])]

        lip_feature, emotion_feature = get_visual_feature_from_GRID(speaker[i], name[i], 
                                                                    instruct_emotion = INSTRUCT_EMOTION,
                                                                    instruct_emotion_mode = 'GRID2GRID',
                                                                    instruct_duration = INSTRUCT_DURATION,
                                                                    instruct_duration_mode = instruct_duration_mode)
        gt_wav_path = "{}/{}/{}.wav".format(wav_path, speaker[i], name[i])

        ref_s, mel_len = compute_style(gt_wav_path)
        wav, dur_pred = inference(phoneme_idx, ref_s, alpha=alpha, beta=beta, embedding_scale=1, 
                        mel_len=mel_len*1.5, emotion_feature=emotion_feature, lip_feature=lip_feature)
        sf.write('{}/wav_pred_{}.wav'.format(save_path, name[i]), wav, 24000)
        duration_pred.append({"path": "{}/{}.wav".format(speaker[i], name[i]), "prediction":dur_pred.tolist()})


