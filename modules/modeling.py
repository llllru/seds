from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import copy
import torch
import pickle as pkl
from torch import nn

from modules.until_module import PreTrainedModel, AllGather, CrossEn, KL
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip
from modules.module_fusionencoder import MLP_feature_fusion, Gloss_Fusion_Transformer
import  torch.nn.functional as F
from modules.module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from modules.modeling_signbert import init_sign_model

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None
        self.distributed = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None,distributed=False, type_vocab_size=2, *inputs, **kwargs):
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)
        model.distributed=distributed
        ## ===> Initialization trick [HARD CODE]

        ## <=== End of initialization trick

        if state_dict is not None:
            model_state_dict = model.state_dict()
            for key in ["clip.visual.conv1.weight", "clip.visual.positional_embedding"]:
                if key in state_dict and key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
                    state_dict.pop(key)
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class GatedFusion(nn.Module):
    def __init__(self, pose_dim, i3d_dim, hidden_dim, dropout=0.1):
        super(GatedFusion, self).__init__()
        self.pose_proj = nn.Linear(pose_dim, hidden_dim)
        self.i3d_proj = nn.Linear(i3d_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pose_feat, i3d_feat):
        pose_feat = self.pose_proj(pose_feat)
        i3d_feat = self.i3d_proj(i3d_feat)
        gate = torch.sigmoid(self.gate(torch.cat([pose_feat, i3d_feat], dim=-1)))
        fused = gate * i3d_feat + (1.0 - gate) * pose_feat
        return self.norm(self.dropout(fused))


class PartAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(PartAttention, self).__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, part_features):
        # part_features: [B, T, P, H]
        weight = torch.softmax(self.score(part_features), dim=2)
        return torch.sum(weight * part_features, dim=2)

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        self._stage_one = True
        self._stage_two = False
        self.signbert_have = task_config.signbert
        self.fusion_type = task_config.fusion_type
        self.freeze_exfusion = task_config.freeze_exfusion
        self.dual_mix = task_config.dual_mix
        self.mix_design = task_config.mix_design

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32
        vision_layers=self.task_config.visual_num_hidden_layers
        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t pose_dim: {}".format(task_config.pose_dim))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,feature_len=task_config.feature_len, input_size=task_config.pose_dim,
            linear_patch=self.linear_patch
        ).float()
        self.aug_choose=task_config.aug_choose
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        
        if self.signbert_have:
            self.signbert = init_sign_model(args=task_config)

        self.use_i3d_local_features = getattr(task_config, "use_i3d_local_features", False)
        i3d_dim = getattr(task_config, "video_dim", 1024)
        part_pose_dim = getattr(task_config, "hidden_dim", 512)
        fusion_dim = getattr(task_config, "hidden_dim", 512)
        self.left_i3d_fusion = GatedFusion(part_pose_dim, i3d_dim, fusion_dim, task_config.dropout)
        self.right_i3d_fusion = GatedFusion(part_pose_dim, i3d_dim, fusion_dim, task_config.dropout)
        self.body_i3d_fusion = GatedFusion(part_pose_dim, i3d_dim, fusion_dim, task_config.dropout)
        self.part_attention = PartAttention(fusion_dim)
        self.i3d_fusion_output = nn.Linear(fusion_dim, task_config.pose_dim)

        self.loss_fct = CrossEn()

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, right_batch, left_batch, body_batch, input_ids_aug=None, attention_mask_aug=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        input_ids_aug = input_ids_aug.view(-1, input_ids.shape[-1])

        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        attention_mask_aug=attention_mask_aug.view(-1, attention_mask.shape[-1])

        sequence_hidden, text_mask, visual_hidden_pose, video_mask, sequence_hidden_aug, text_mask_aug= self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         right_batch, left_batch, body_batch, shaped=True, input_ids_aug=input_ids_aug, attention_mask_aug=attention_mask_aug)

        if self.training:
            loss = 0.

            if self.sim_header == "Filip":
                I2T_sim_pose, T2I_sim_pose = self.get_similarity_logits(sequence_hidden, visual_hidden_pose, text_mask,
                                                                     video_mask,
                                                                     shaped=True, loose_type=self.loose_type,sequence_hidden_aug=sequence_hidden_aug,text_mask_aug=text_mask_aug)

                sim_loss1_pose=self.loss_fct(I2T_sim_pose)*self.dual_mix+ self.loss_fct(I2T_sim_pose.T)*(1-self.dual_mix)
                sim_loss2_pose=self.loss_fct(T2I_sim_pose.T)*self.dual_mix+self.loss_fct(T2I_sim_pose)*(1-self.dual_mix)
                sim_loss_pose = (sim_loss1_pose + sim_loss2_pose) / 2

                loss += sim_loss_pose

            return loss, sim_loss_pose
        else:
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False,get_hidden=True):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        if self.sim_header=='Filip'and get_hidden==True:
            text_mask,sequence_hidden = self.clip.encode_text(input_ids,return_hidden=True)  # B,512

            sequence_hidden=sequence_hidden.float()
            return text_mask, sequence_hidden

        return sequence_hidden

    def get_sign_output(self, right_batch, left_batch, body_batch):
        
        clips_start = body_batch['clips_start']
        clip_mask = body_batch['mask']
        batch_num, feature_len = clips_start.size()
        slide_windows = self.task_config.slide_windows
        right_i3d = right_batch.get('i3d')
        left_i3d = left_batch.get('i3d')
        body_i3d = body_batch.get('i3d')
        
        pose_all = {}
        pose_all['right'] = right_batch['pose']
        pose_all['left'] = left_batch['pose']
        pose_all['body'] = body_batch['pose']

        pose_all = self.signbert.gcn_emb(pose_all)

        if self.use_i3d_local_features and right_i3d is not None and left_i3d is not None and body_i3d is not None:
            pose_final_new = self.get_i3d_fused_sign_output(
                pose_all,
                clips_start,
                slide_windows,
                left_i3d,
                right_i3d,
                body_i3d,
            )
            return pose_final_new, clip_mask

        del right_batch, left_batch, body_batch
        torch.cuda.empty_cache()

        batch_num, seq_length, feat_dim = pose_all['feat'].size()
        pose_final_new = torch.zeros((batch_num, feature_len, slide_windows, feat_dim)).to(device=pose_all['feat'].device, dtype=pose_all['feat'].dtype)
        # [B, 64, 16, 1536]表示把整段pose序列切成clip窗口后的pose特征。
        for i in range(batch_num):
            for j in range(feature_len):
                if clips_start[i, j] != -1:
                    assert clip_mask[i, j+1] == 0
                    pose_final_new[i, j, :, :] = pose_all['feat'][i, clips_start[i,j]:clips_start[i,j]+slide_windows, :]
                else:
                    assert clip_mask[i, j+1] == 1

        del pose_all
        torch.cuda.empty_cache()
        

        pose_final_new = pose_final_new.reshape(batch_num*feature_len, slide_windows, feat_dim)
        pose_final_new = self.signbert.sign_conv(pose_final_new)
        pose_final_new = pose_final_new.reshape(batch_num, feature_len, slide_windows, feat_dim)
        pose_final_new = torch.mean(pose_final_new, dim=-2)
        
        pose_final_new = pose_final_new.permute(0, 2, 1).unsqueeze(-1)
        
        return pose_final_new, clip_mask

    def collect_part_clip_features(self, part_feat, clips_start, slide_windows):
        batch_num, feature_len = clips_start.size()
        feat_dim = part_feat.size(-1)
        part_clip_feat = torch.zeros((batch_num, feature_len, feat_dim)).to(
            device=part_feat.device,
            dtype=part_feat.dtype,
        )
        for i in range(batch_num):
            for j in range(feature_len):
                if clips_start[i, j] != -1:
                    start = int(clips_start[i, j].item())
                    part_clip_feat[i, j] = torch.mean(part_feat[i, start:start + slide_windows, :], dim=0)
        return part_clip_feat

    def pad_i3d_to_feature_len(self, i3d_feat, feature_len):
        if i3d_feat.size(1) == feature_len:
            return i3d_feat
        padded = torch.zeros((i3d_feat.size(0), feature_len, i3d_feat.size(-1))).to(
            device=i3d_feat.device,
            dtype=i3d_feat.dtype,
        )
        valid_len = min(feature_len, i3d_feat.size(1))
        padded[:, :valid_len, :] = i3d_feat[:, :valid_len, :]
        return padded

    def get_i3d_fused_sign_output(self, pose_all, clips_start, slide_windows, left_i3d, right_i3d, body_i3d):
        feature_len = clips_start.size(1)
        left_pose = self.collect_part_clip_features(pose_all['left_feat'], clips_start, slide_windows)
        right_pose = self.collect_part_clip_features(pose_all['right_feat'], clips_start, slide_windows)
        body_pose = self.collect_part_clip_features(pose_all['body_feat'], clips_start, slide_windows)

        left_i3d = self.pad_i3d_to_feature_len(left_i3d, feature_len)
        right_i3d = self.pad_i3d_to_feature_len(right_i3d, feature_len)
        body_i3d = self.pad_i3d_to_feature_len(body_i3d, feature_len)

        left_fused = self.left_i3d_fusion(left_pose, left_i3d)
        right_fused = self.right_i3d_fusion(right_pose, right_i3d)
        body_fused = self.body_i3d_fusion(body_pose, body_i3d)

        part_features = torch.stack([left_fused, right_fused, body_fused], dim=2)
        fused = self.part_attention(part_features)
        fused = self.i3d_fusion_output(fused)
        return fused.permute(0, 2, 1).unsqueeze(-1)

    def get_visual_output(self, right_batch, left_batch, body_batch, shaped=True, get_hidden=True):
        
        video_pose, video_mask = self.get_sign_output(right_batch, left_batch, body_batch)

        bs_pair = video_mask.size(0)
        video_frame=1

        if self.sim_header == 'Filip' and get_hidden==True:
            _, visual_hidden_pose = self.clip.encode_image(video_pose, return_hidden=True, video_mask=video_mask, video_frame=video_frame)
            visual_hidden_pose=visual_hidden_pose.float()

            visual_hidden_pose = visual_hidden_pose.view(bs_pair, -1, visual_hidden_pose.size(-1))

        else:
            ValueError("no such sim_header!!!")

        return video_mask, visual_hidden_pose

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, right_batch, left_batch, body_batch, shaped=False, input_ids_aug=None, attention_mask_aug=None):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        
        text_mask, sequence_hidden = self.get_sequence_output(input_ids,  token_type_ids, attention_mask, shaped=False)
        text_mask_aug, sequence_hidden_aug = self.get_sequence_output(input_ids_aug,  token_type_ids, attention_mask_aug, shaped=False)

        video_mask, visual_hidden_pose = self.get_visual_output(right_batch, left_batch, body_batch, shaped=True)

        return sequence_hidden, text_mask, visual_hidden_pose, video_mask, sequence_hidden_aug, text_mask_aug
    
    def flip_similarity_softmax(self, sequence_output, visual_hidden_pose, attention_mask, video_mask, sim_header="meanP", pad_type=1, sequence_hidden_aug=None, text_mask_aug=None):
        if self.training and self.distributed:
            visual_hidden_pose = allgather(visual_hidden_pose, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            sequence_hidden_aug = allgather(sequence_hidden_aug, self.task_config)
            attention_mask = allgather(attention_mask, self.task_config)
            text_mask_aug = allgather(text_mask_aug, self.task_config)
            
            torch.distributed.barrier()

        video_mask = (video_mask == 0)
        attention_mask = (attention_mask==1)
        text_mask_aug = (text_mask_aug==1)

        visual_hidden_pose = visual_hidden_pose / visual_hidden_pose.norm(dim=-1, keepdim=True)
        visual_hidden_pose = visual_hidden_pose.squeeze(1)

        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        sequence_output = sequence_output.squeeze(1)

        batch_size, v_len=visual_hidden_pose.shape[0],visual_hidden_pose.shape[1]
        batch_size_t, t_len=sequence_output.shape[0],sequence_output.shape[1]

        sequence_hidden_aug = sequence_hidden_aug / sequence_hidden_aug.norm(dim=-1, keepdim=True)
        sequence_hidden_aug = sequence_hidden_aug.squeeze(1)

        logit_scale = self.clip.logit_scale.exp()
        
        i2t_sim_pose=torch.einsum("ais, bjs->abij", [visual_hidden_pose, sequence_output])
        i2t_sim_aug_pose=torch.einsum("ais, bjs->abij", [visual_hidden_pose, sequence_hidden_aug])

        after_softmax_i2t_pose = torch.nansum(i2t_sim_pose * torch.softmax(i2t_sim_pose/0.07, dim=3), dim=3)
        video_mask_extend=video_mask.unsqueeze(1).repeat(1,batch_size_t,1)
        after_softmax_i2t_pose[~video_mask_extend]=0
        I2T_sim_pose = logit_scale*torch.nansum(after_softmax_i2t_pose, dim=-1)/torch.sum(video_mask_extend,dim=-1)

        after_softmax_t2i_pose = torch.nansum(i2t_sim_aug_pose * torch.softmax(i2t_sim_aug_pose/0.07, dim=2), dim=2)
        text_mask_extend2=text_mask_aug.unsqueeze(0).repeat(batch_size,1,1)
        after_softmax_t2i_pose[~text_mask_extend2]=0
        T2I_sim_pose = logit_scale*torch.nansum(after_softmax_t2i_pose*text_mask_extend2, dim=-1)/torch.sum(text_mask_extend2,dim=-1)

        return I2T_sim_pose, T2I_sim_pose

    def get_similarity_logits(self, sequence_output, visual_hidden_pose, attention_mask, video_mask, shaped=False, loose_type=False,is_train=True,sequence_hidden_aug=None,text_mask_aug=None):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if sequence_hidden_aug==None:
            sequence_hidden_aug=sequence_output
            text_mask_aug=attention_mask

        if video_mask[0][0] == 1:
            ValueError("the video_mask[0][0] == 1")
        
        if self.sim_header=='Filip' and is_train==True:
            
            I2T_sim_pose, T2I_sim_pose = self.flip_similarity_softmax(sequence_output, visual_hidden_pose, attention_mask, video_mask,
                                                     sim_header=self.sim_header,sequence_hidden_aug=sequence_hidden_aug,text_mask_aug=text_mask_aug)

            return I2T_sim_pose, T2I_sim_pose
        
        return None, None
