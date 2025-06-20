import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate

import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

import utils.paramUtil as paramUtil
from dataset.dataset_VQ_bodypart import whole2parts, parts2whole

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 차원 계산 - 짝수 차원만 사용
        div_term_dim = d_model if d_model % 2 == 0 else d_model - 1
        div_positions = torch.arange(0, div_term_dim, 2).float()
        
        # 10000^(2i/d_model) 계산
        div_term = torch.pow(10000, div_positions / d_model)
        
        # sin, cos 적용 - 차원 크기 체크
        pe[:, 0:div_term_dim:2] = torch.sin(position / div_term)
        pe[:, 1:div_term_dim:2] = torch.cos(position / div_term)
        
        # 홀수 차원인 경우 마지막 차원 처리
        if d_model % 2 != 0:
            last_position = torch.arange(d_model-1, d_model).float()
            last_div_term = torch.pow(10000, last_position / d_model)
            pe[:, -1] = torch.sin(position.squeeze(-1) / last_div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [seq_len, feature_dim]
        """
        return x + self.pe[:x.size(0), :x.size(1)]



def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, is_test, w_vectorizer, feat_bias=5, max_text_len=20, unit_length=4, print_warning=False,
                 is_train=False):
        
        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        self.print_warning = print_warning

        if dataset_name == 't2m':
            self.data_root = '/local_datasets/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
            self.meta_dir = 'checkpoints/t2m/Decomp_SP001_SM001_H512/meta'
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196
            kinematic_chain = paramUtil.kit_kinematic_chain
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))

        if is_train:
            assert is_test is None
            split_file = pjoin(self.data_root, 'train.txt')
        else:
            if is_test:
                split_file = pjoin(self.data_root, 'test.txt')
            else:
                split_file = pjoin(self.data_root, 'val.txt')

        min_motion_len = 40 if self.dataset_name =='t2m' else 24
        # min_motion_len = 64

        joints_num = self.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        print(len(id_list))

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))

                # debug
                if np.isnan(motion).sum() > 0:
                    if self.print_warning:
                        print('Detected NaN in Dataset, initialization stage!')
                        print('npy name:', pjoin(self.motion_dir, name + '.npy'))
                    continue


                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    if self.print_warning:
                        print('Skip the motion:', name, '. motion length is shorter than min_motion_len or greater than 200.')
                    continue

                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():

                        # print(line)

                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*fps) : int(to_tag*fps)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue

                                # assign new name
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name

                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text': [text_dict]}

                                # print("Ready to append list", new_name, len(n_motion))

                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                if self.print_warning:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))

            except Exception as e:
                if self.print_warning:
                    print('Unable to load:', name)
                    print(e)
                # pass

        # print(len(new_name_list), len(length_list))

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def whole2parts(self, motion, mode='t2m'):
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = whole2parts(motion, mode)
        return [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]

    def parts2whole(self, parts, mode='t2m', shared_joint_rec_mode='Avg'):
        rec_data = parts2whole(parts, mode, shared_joint_rec_mode)
        return rec_data

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        # data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length

        # motion length is dynamic at inference stage, not fixed.
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        # debug
        if np.isnan(motion).sum() > 0:
            print('Detected NaN in Dataset, before preprocess!')

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # zero padding, along the nframes dimension
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        parts = self.whole2parts(motion, mode=self.dataset_name)  # [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts
        # 각 신체 부위의 feature 차원 확인 및 positional encoding 적용
        # numpy 배열을 torch tensor로 변환
        Root = torch.from_numpy(Root).float()
        R_Leg = torch.from_numpy(R_Leg).float()
        L_Leg = torch.from_numpy(L_Leg).float()
        Backbone = torch.from_numpy(Backbone).float()
        R_Arm = torch.from_numpy(R_Arm).float()
        L_Arm = torch.from_numpy(L_Arm).float()

        # feature 차원 가져오기
        feature_dim = Root.shape[1]

        # PositionalEncoding 인스턴스 생성
        pos_encoder = PositionalEncoding(d_model=feature_dim, max_len=self.max_motion_length)

        # 각 부위에 positional encoding 적용
        Root = pos_encoder(Root)
        R_Leg = pos_encoder(R_Leg)
        L_Leg = pos_encoder(L_Leg)
        Backbone = pos_encoder(Backbone)
        R_Arm = pos_encoder(R_Arm)
        L_Arm = pos_encoder(L_Arm)

        # tensor를 다시 numpy로 변환 (필요한 경우)
        Root = Root.numpy()
        R_Leg = R_Leg.numpy()
        L_Leg = L_Leg.numpy()
        Backbone = Backbone.numpy()
        R_Arm = R_Arm.numpy() 
        L_Arm = L_Arm.numpy()

        # 업데이트된 parts 리스트 생성
        parts = [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]

        # debug
        if np.isnan(motion).sum() > 0:
            print('Detected NaN in Dataset, after preprocess!')


        # return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name, parts




def DATALoader(dataset_name, is_test,
                batch_size, w_vectorizer,
                num_workers = 8, unit_length = 4,
               is_train=False) :
    
    val_loader = torch.utils.data.DataLoader(
        Text2MotionDataset(dataset_name, is_test, w_vectorizer, unit_length=unit_length, is_train=is_train),
                                              batch_size,
                                              shuffle = True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last = True)
    return val_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x