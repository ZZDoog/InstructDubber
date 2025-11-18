import os
import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# instruct_emotion_path = 'Instruct_Emotion_Audio'
# # instruct_emotion_path = 'Instruct_Emotion_Raw'
# # instruct_emotion_path = 'Instruct_Emotion_Distill_428'

def get_visual_feature_from_V2C(speaker, name, instruct_emotion=True, 
                                instruct_emotion_mode='Instruct_Emotion_Audio', 
                                instruct_duration = False,
                                instruct_duration_mode='Instruct_Lip_Raw'):
    
    if instruct_duration:
        lip_path = os.path.join(
            "/data1/home/zhangzhedong/preprocessed_data/Denoise_version2_all_feature_V2C",
            # "Instruct_Lip_Raw_Qwen3_4B",
            instruct_duration_mode,
            "{}-face-{}.npy".format(speaker, name),
        )

    else: 
        lip_path = os.path.join(
            "/data1/home/zhangzhedong/preprocessed_data/Denoise_version2_all_feature_V2C",
            "extrated_embedding_V2C_gray",
            "{}-face-{}.npy".format(speaker, name),
        )
    lip_feature = torch.from_numpy(np.load(lip_path)).float().to(device)

    if instruct_emotion:
        emotion_path = os.path.join(
            "/data1/home/zhangzhedong/preprocessed_data/Denoise_version2_all_feature_V2C",
            # "Instruct_Emotion_Distill_428",
            # "Instruct_Emotion_Audio",
            instruct_emotion_mode,
            "{}-feature-{}.npy".format(speaker, name),
        )
        emotion_feature = torch.from_numpy(np.load(emotion_path)).float().to(device)
    else:
        emotion_path = os.path.join(
            "/data1/home/zhangzhedong/preprocessed_data/Denoise_version2_all_feature_V2C",
            "VA_feature",
            "{}-feature-{}.npy".format(speaker, name),
        )
        emotion_feature = torch.from_numpy(np.load(emotion_path)).float().to(device)

    return lip_feature, emotion_feature

def get_visual_feature_from_GRID(speaker, name, instruct_emotion=True, 
                                 instruct_emotion_mode='Instruct_Emotion_Audio', 
                                 instruct_duration = False,
                                 instruct_duration_mode='Instruct_Lip_Raw'):
    if instruct_duration:
        lip_path = os.path.join(
            "/data1/home/zhangzhedong/preprocessed_data/Grid_Wav_22050_Abs_Feature",
            # "Instruct_Lip_Raw_Qwen3_4B",
            instruct_duration_mode,
            "{}-face-{}.npy".format(speaker, name.split('-')[-1]),
        )

    else:
        lip_path = os.path.join(
            "/data1/home/zhangzhedong/preprocessed_data/Grid_Wav_22050_Abs_Feature",
            "extrated_embedding_Grid_152_gray",
            "{}-face-{}.npy".format(speaker, name.split('-')[-1]),
        )
    lip_feature = torch.from_numpy(np.load(lip_path)).float().to(device)

    if instruct_emotion:
        emotion_path = os.path.join(
                "/data1/home/zhangzhedong/preprocessed_data/Grid_Wav_22050_Abs_Feature",
                # "Instruct_Emotion_Distill_428",
                # "Instruct_Emotion_Audio",
                instruct_emotion_mode,
                "{}-feature-{}.npy".format(speaker, name.split('-')[-1]),
            )
        emotion_feature = torch.from_numpy(np.load(emotion_path)).float().to(device)
    else:
        emotion_path = os.path.join(
            "/data1/home/zhangzhedong/preprocessed_data/Grid_Wav_22050_Abs_Feature",
            "VA_feature",
            "{}-feature-{}.npy".format(speaker, name.split('-')[-1]),
        )
        emotion_feature = torch.from_numpy(np.load(emotion_path)).float().to(device)

    return lip_feature, emotion_feature

def get_visual_feature_from_Chem(speaker, name, instruct_emotion=True, 
                                instruct_emotion_mode='Instruct_Emotion_Audio', 
                                instruct_duration = False,
                                instruct_duration_mode='Instruct_Lip_Raw'):
    if instruct_duration:
        lip_path = os.path.join(
            "/data1/home/zhangzhedong/preprocessed_data/Chem_Feature2025",
            # "Instruct_Lip_Raw_Qwen3_4B",
            instruct_duration_mode,
            "lipmotion-{}.npy".format(name),
        )

    else:
        lip_path = os.path.join(
            "/data1/home/zhangzhedong/preprocessed_data/Chem_Feature2025",
            "lipmotion",
            "lipmotion-{}.npy".format(name),
        )

    lip_feature = torch.from_numpy(np.load(lip_path)).float().to(device)

    if instruct_emotion:
        emotion_path = os.path.join(
            "/data1/home/zhangzhedong/preprocessed_data/Chem_Feature2025",
            # "Instruct_Emotion_Distill_428",
            # "Instruct_Emotion_Audio",
            instruct_emotion_mode,
            "{}-feature-{}.npy".format(speaker, name),
        )
        emotion_feature = torch.from_numpy(np.load(emotion_path)).float().to(device)
    else:
        emotion_path = os.path.join(
            "/data1/home/zhangzhedong/preprocessed_data/Chem_Feature2025",
            "VA_feature",
            "{}-feature-{}.npy".format(speaker, name),
        )
        emotion_feature = torch.from_numpy(np.load(emotion_path)).float().to(device)

    return lip_feature, emotion_feature

    