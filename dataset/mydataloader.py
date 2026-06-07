import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import json
import re
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import random
def split_sentence(sentence):
    # Translated comment.
    pattern = r'[a-zA-Z0-9]+'
    words = re.findall(pattern, sentence.replace('\n', ' '))
    words =[w.lower() for w in words]
    return words

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])


class PandasData(Dataset):
    def __init__(self, base_dir,csv_file,caption_json,vocab,max_len=50,frame=30):
        self.base_dir = base_dir
        self.max_len = max_len
        self.video_dir = "videos"
        self.frame = frame
        self.data = pd.read_csv(os.path.join(base_dir,csv_file),header=None).values
        self.data = [x[0] for x in self.data]
        with open(os.path.join(base_dir,caption_json), 'r', encoding='utf-8') as file:
            self.caption_list = json.load(file)
        self.cap = {}
        for caption in self.caption_list:
            self.cap[caption["video"]] = caption["short"][0]
        with open(os.path.join(base_dir,vocab), 'r', encoding='utf-8') as file:
            self.vocab = json.load(file)

    def __len__(self):
        return len(self.data)

    def extract_and_transform(self, video_path, target_frame_count=30):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("The padded sequence exceeds the target length.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = int(total_frames / fps)

        transformed_frames = []

        for i in range(min(duration_seconds, target_frame_count)):
            middle_frame_index = int((i + 0.5) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                tensor = transform(pil_image)
                transformed_frames.append(tensor)
            else:
                break

        cap.release()

        # Translated comment.
        if transformed_frames:
            black_tensor = torch.zeros_like(transformed_frames[0])
        else:
            black_tensor = transform(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))

        while len(transformed_frames) < target_frame_count:
            transformed_frames.append(black_tensor.clone())

        final_tensor = torch.stack(transformed_frames[:target_frame_count])  # shape: [30, 3, 224, 224]
        return final_tensor

    def pad_list_to_tensor(self,original_list, target_length):
        padded = [1] + original_list + [2]

        padding_needed = target_length - len(padded)
        if padding_needed < 0:
            raise ValueError("The padded sequence exceeds the target length.")

        padded += [0] * padding_needed
        return torch.tensor(padded, dtype=torch.long)
    def pad_list_to_tgt(self,original_list, target_length):
        padded = original_list + [2]

        padding_needed = target_length - len(padded)
        if padding_needed < 0:
            raise ValueError("The padded sequence exceeds the target length.")

        padded += [0] * padding_needed
        return torch.tensor(padded, dtype=torch.long)
    def __getitem__(self, idx):
        vn = self.data[idx]
        src_dir=os.path.join(self.base_dir,self.video_dir,vn+".mp4")
        src = self.extract_and_transform(src_dir,self.frame)

        caption=self.cap[vn]
        words=split_sentence(caption)
        word_idx=[self.vocab[w] for w in words]
        tgt_in = self.pad_list_to_tensor(word_idx,self.max_len)
        tgt_out = self.pad_list_to_tgt(word_idx,self.max_len)
        ret ={
            "visual":src,
            'text':tgt_in,
            'target':tgt_out,
            'vn':vn
        }
        return ret

class MsrvttData(Dataset):
    def __init__(self, base_dir,csv_file,caption_json,tokenizer,max_len=50,frame=30):
        self.base_dir = base_dir
        self.max_len = max_len
        self.video_dir = "videos"
        self.frame = frame
        self.tokenizer = tokenizer
        self.data = pd.read_csv(os.path.join(base_dir, csv_file), header=None, dtype=str).values
        self.data = [x[0] for x in self.data]
        with open(os.path.join(base_dir,caption_json), 'r', encoding='utf-8') as file:
            self.cap = json.load(file)

    def __len__(self):
        return len(self.data)

    def extract_and_transform(self, video_path, target_frame_count=30):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("The padded sequence exceeds the target length.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        transformed_frames = []
        nums = np.linspace(1, total_frames, target_frame_count, dtype=int)
        for i in nums:
            middle_frame_index = i
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                tensor = transform(pil_image)
                transformed_frames.append(tensor)
            else:
                break

        cap.release()

        # Translated comment.
        if transformed_frames:
            black_tensor = torch.zeros_like(transformed_frames[0])
        else:
            black_tensor = transform(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))

        while len(transformed_frames) < target_frame_count:
            transformed_frames.append(black_tensor.clone())

        final_tensor = torch.stack(transformed_frames[:target_frame_count])  # shape: [30, 3, 224, 224]
        return final_tensor

    def pad_list_to_tensor(self,original_list, target_length):
        padded =  original_list
        padding_needed = target_length - len(padded)
        if padding_needed < 0:
            raise ValueError("The padded sequence exceeds the target length.")
        padded += [0] * padding_needed
        return torch.tensor(padded, dtype=torch.long)

    def pad_list_to_tgt(self,original_list, target_length,frame):
        padded = original_list
        padding_needed = target_length - len(padded)
        if padding_needed < 0:
            raise ValueError("The padded sequence exceeds the target length.")
        padded += [0] * padding_needed
        padded = torch.tensor(padded, dtype=torch.long)
        mask = (padded != 0).float()
        mask = torch.cat((torch.ones(frame), mask), dim=0)  # adding prefix mask
        return padded,mask
    def __getitem__(self, idx):
        vn = self.data[idx]
        src_dir=os.path.join(self.base_dir,self.video_dir,vn+".mp4")
        src = self.extract_and_transform(src_dir,self.frame)
        caption=self.cap[vn]
        caption = self.tokenizer.encode(caption)
        tgt_in,mask = self.pad_list_to_tgt(caption,self.max_len, self.frame )
        ret ={
            "visual":src,
            'text':tgt_in,
            'mask':mask,
            'vn':vn
        }
        return ret

class MsrvttData_feature(Dataset):
    def __init__(self, base_dir,feature_dir,csv_file,caption_json,tokenizer,max_len=100,frame=8):
        self.base_dir = base_dir
        self.max_len = max_len
        self.feature_dir = feature_dir
        self.sg_dir = "scene"
        self.tokenizer = tokenizer
        self.frame = frame
        self.data = pd.read_csv(os.path.join(base_dir, csv_file), header=None, dtype=str).values
        self.data = [x[0] for x in self.data]
        with open(os.path.join(base_dir,caption_json), 'r', encoding='utf-8') as file:
            self.cap = json.load(file)

    def __len__(self):
        return len(self.data)

    def pad_list_to_tensor(self,original_list, target_length):
        padded = original_list

        padding_needed = target_length - len(padded)
        if padding_needed < 0:
            raise ValueError("The padded sequence exceeds the target length.")

        padded += [0] * padding_needed
        return torch.tensor(padded, dtype=torch.long)

    def pad_list_to_tgt(self,original_list, target_length,frame):
        padded = original_list
        target = original_list + [50256]
        padding_needed = target_length - len(padded)
        if padding_needed < 0:
            raise ValueError("The padded sequence exceeds the target length.")
        padded += [50256] * padding_needed
        target += [0] * padding_needed
        padded = torch.tensor(padded, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        mask =  (target != 0).long()
        return padded,target,mask

    def __getitem__(self, idx):
        vn = self.data[idx]
        src_dir=os.path.join(self.base_dir,self.feature_dir,vn+".pt")
        graph_dir = os.path.join(self.base_dir, self.sg_dir, vn + ".npy")
        graph = np.load(graph_dir)
        graph = torch.from_numpy(graph)
        graph = graph.float()
        visual = torch.load(src_dir,map_location='cpu').float()

        caption=self.cap[vn]
        if isinstance(caption, list):  # Translated comment.
            caption = random.choice(caption)
        caption = self.tokenizer.encode(caption)
        tgt_in ,target, mask = self.pad_list_to_tgt(caption,self.max_len,self.frame)
        ret ={
            "visual":visual,
            'text':tgt_in,
            'graph':graph,
            'target':target,
            'mask':mask,
            'vn':vn
        }
        # [-1 -1 -1 1 2 3 4 0 -1 -1 -1 -1]
        # [eos eos eos 1 2 3 4 0 eos eos eos]

        # frames(n) + sentence(len+pad)
        # pad*n  + sentence
        # mask
        return ret

class Panda_feature(Dataset):
    def __init__(self, base_dir, feature_dir, csv_file, caption_json, tokenizer,cap_type='short',max_len=50, frame=8):
        self.base_dir = base_dir
        self.max_len = max_len
        self.feature_dir = feature_dir
        self.sg_dir = "scene"
        self.tokenizer = tokenizer
        self.frame = frame
        self.data = pd.read_csv(os.path.join(base_dir,csv_file),header=None).values
        self.data = [x[0] for x in self.data]
        with open(os.path.join(base_dir,caption_json), 'r', encoding='utf-8') as file:
            self.caption_list = json.load(file)
        self.cap = {}
        for caption in self.caption_list:
            self.cap[caption["video"]] = caption[cap_type][0]

    def __len__(self):
        return len(self.data)

    def pad_list_to_tgt(self,original_list, target_length,frame):
        padded = original_list
        target = original_list + [50256]
        padding_needed = target_length - len(padded)
        if padding_needed < 0:
            raise ValueError("The padded sequence exceeds the target length.")
        padded += [50256] * padding_needed
        target += [0] * padding_needed
        padded = torch.tensor(padded, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        mask =  (target != 0).long()
        return padded,target,mask

    def __getitem__(self, idx):
        vn = self.data[idx]
        src_dir=os.path.join(self.base_dir,self.feature_dir,vn+".pt")
        graph_dir = os.path.join(self.base_dir, self.sg_dir, vn + ".npy")
        graph = np.load(graph_dir)
        graph = torch.from_numpy(graph)
        graph = graph.float()
        visual = torch.load(src_dir,map_location='cpu').float()
        caption=self.cap[vn]
        caption = self.tokenizer.encode(caption)
        tgt_in ,target, mask = self.pad_list_to_tgt(caption,self.max_len,self.frame)
        ret ={
            "visual":visual,
            'text':tgt_in,
            'graph':graph,
            'target':target,
            'mask':mask,
            'vn':vn
        }
        return ret

if __name__ == "__main__":
    # train_dataset = PandasData("panda",'train_data.csv',caption_json="20250501.json",vocab="vocab.json")
    from transformers import GPT2Tokenizer
    gpt2_model_path = "E:\pycharmProject\clipcap\pretrained\gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
    # train_dataset = MsrvttData_feature('msrvtt','vitb32','test_data.csv',
    #                                    'caption_all.json',tokenizer,max_len=100,frame=8)
    # train_dataset = MsrvttData_feature('msvd','vit_1fps','test_data.csv',
    #                                    'caption.json',tokenizer,max_len=50,frame=8)
    train_dataset = Panda_feature('panda','vitb32','train_data.csv',
                                        '20250808.json',tokenizer,cap_type='detailed',max_len=75,frame=8)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for batch in train_loader:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

    # train_dataset = MsrvttData('msrvtt','test_data.csv',caption_json="caption.json",tokenizer=tokenizer,frame=8)
    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # for batch in train_loader:
    #     for key in batch:
    #         if isinstance(batch[key], torch.Tensor):
    #             batch[key] = batch[key].to(device)
    #     break




