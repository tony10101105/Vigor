import torch
from torch import nn
import numpy as np
from .utils import get_siamese_features, get_mlp_head
import math
try:
    from . import PointNetPP
except ImportError:
    PointNetPP = None

from transformers import BertModel, BertConfig
from referit3d.models import MLP

from .encoder_decoder_layers import RefEcoderLayer

class ReferIt3DNet_transformer(nn.Module):

    def __init__(self,
                 args,
                 n_obj_classes,
                 class_name_tokens,
                 ignore_index):

        super().__init__()

        self.bert_pretrain_path = args.bert_pretrain_path

        self.view_number = args.view_number
        self.rotate_number = args.rotate_number

        self.label_lang_sup = args.label_lang_sup
        self.aggregate_type = args.aggregate_type

        self.encoder_layer_num = args.encoder_layer_num
        self.decoder_layer_num = args.decoder_layer_num
        self.decoder_nhead_num = args.decoder_nhead_num

        self.object_dim = args.object_latent_dim
        self.inner_dim = args.inner_dim
        
        self.dropout_rate = args.dropout_rate
        self.lang_cls_alpha = args.lang_cls_alpha
        self.obj_cls_alpha = args.obj_cls_alpha

        self.object_encoder = PointNetPP(sa_n_points=[32, 16, None],
                                        sa_n_samples=[[32], [32], [None]],
                                        sa_radii=[[0.2], [0.4], [None]],
                                        sa_mlps=[[[3, 64, 64, 128]],
                                                [[128, 128, 128, 256]],
                                                [[256, 256, self.object_dim, self.object_dim]]])

        self.language_encoder = BertModel.from_pretrained(self.bert_pretrain_path)
        self.language_encoder.encoder.layer = BertModel(BertConfig()).encoder.layer[:self.encoder_layer_num]
    
        # Classifier heads
        self.language_clf = get_mlp_head(self.inner_dim, self.inner_dim, n_obj_classes, dropout=self.dropout_rate)
        if args.lang_multilabel:
            self.anchor_clf = get_mlp_head(self.inner_dim, self.inner_dim, 485, dropout=self.dropout_rate)
        self.object_language_clf = get_mlp_head(self.inner_dim, self.inner_dim, 1, dropout=self.dropout_rate)

        if not self.label_lang_sup:
            self.obj_clf = MLP(self.inner_dim, [self.object_dim, self.object_dim, n_obj_classes], dropout_rate=self.dropout_rate)

        self.obj_feature_mapping = nn.Sequential(
            nn.Linear(self.object_dim, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        self.box_feature_mapping = nn.Sequential(
            nn.Linear(4, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        self.class_name_tokens = class_name_tokens

        self.lang_multilabel = args.lang_multilabel
        self.multilabel_pretraining = args.multilabel_pretraining
        self.logit_loss = nn.CrossEntropyLoss()
        self.lang_logits_loss = nn.CrossEntropyLoss()
        if self.lang_multilabel:
            self.anchor_logits_loss = nn.BCEWithLogitsLoss()
        if self.multilabel_pretraining:
            self.ml_feature_constraint_loss = nn.CrossEntropyLoss()
            self.feat_to_multilabel_clf = get_mlp_head(self.inner_dim, self.inner_dim, 1, dropout=self.dropout_rate)

        self.class_logits_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.order_len = args.order_len

        self.refer_encoder = nn.ModuleList()
        for _ in range(self.order_len):
            self.refer_encoder.append(RefEcoderLayer(
                self.inner_dim, n_heads=self.decoder_nhead_num, dim_feedforward=2048,
                dropout=self.dropout_rate, activation="gelu"
            ))

        self.disable_text_loss = args.disable_text_loss
        self.disable_multilabel_loss = args.disable_multilabel_loss

    @torch.no_grad()
    def aug_input(self, input_points, box_infos):
        input_points = input_points.float().to(self.device)
        box_infos = box_infos.float().to(self.device)
        xyz = input_points[:, :, :, :3]
        bxyz = box_infos[:,:,:3] # B,N,3
        B,N,P = xyz.shape[:3]
        rotate_theta_arr = torch.Tensor([i*2.0*np.pi/self.rotate_number for i in range(self.rotate_number)]).to(self.device)
        view_theta_arr = torch.Tensor([i*2.0*np.pi/self.view_number for i in range(self.view_number)]).to(self.device)
        
        # rotation
        if self.training:
            # theta = torch.rand(1) * 2 * np.pi  # random direction rotate aug
            theta = rotate_theta_arr[torch.randint(0,self.rotate_number,(B,))]  # 4 direction rotate aug
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotate_matrix = torch.Tensor([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,1.0]]).to(self.device)[None].repeat(B,1,1)
            rotate_matrix[:, 0, 0] = cos_theta
            rotate_matrix[:, 0, 1] = -sin_theta
            rotate_matrix[:, 1, 0] = sin_theta
            rotate_matrix[:, 1, 1] = cos_theta

            input_points[:, :, :, :3] = torch.matmul(xyz.reshape(B,N*P,3), rotate_matrix).reshape(B,N,P,3)
            bxyz = torch.matmul(bxyz.reshape(B,N,3), rotate_matrix).reshape(B,N,3)
        
        # multi-view
        bsize = box_infos[:,:,-1:]
        boxs=[]
        for theta in view_theta_arr:
            rotate_matrix = torch.Tensor([[math.cos(theta), -math.sin(theta), 0.0],
                                        [math.sin(theta), math.cos(theta),  0.0],
                                        [0.0,           0.0,            1.0]]).to(self.device)
            rxyz = torch.matmul(bxyz.reshape(B*N, 3),rotate_matrix).reshape(B,N,3)
            boxs.append(torch.cat([rxyz,bsize],dim=-1))
        boxs=torch.stack(boxs,dim=1)
        return input_points, boxs

    def compute_basic_loss(self, batch, CLASS_LOGITS, LANG_LOGITS, LOGITS, ANCHOR_LOGITS=None):
        referential_loss = self.logit_loss(LOGITS, batch['target_pos'])
        obj_clf_loss = self.class_logits_loss(CLASS_LOGITS.transpose(2, 1), batch['class_labels'])
        lang_clf_loss = 0
        if not self.disable_text_loss:
            lang_clf_loss = self.lang_logits_loss(LANG_LOGITS, batch['target_class'])
            if ANCHOR_LOGITS is not None:
                lang_clf_loss += self.anchor_logits_loss(ANCHOR_LOGITS, batch['anchor_ind'])
        total_loss = referential_loss + self.obj_cls_alpha * obj_clf_loss + self.lang_cls_alpha * lang_clf_loss
        return total_loss

    def forward(self, batch: dict, epoch=None):
        TOTAL_LOSS = 0
        # batch['class_labels']: GT class of each obj
        # batch['target_class']：GT class of target obj
        # batch['target_pos']: GT id
        self.device = self.obj_feature_mapping[0].weight.device

        ## rotation augmentation and multi_view generation
        obj_points, boxs = self.aug_input(batch['objects'], batch['box_info'])

        B,N,P = obj_points.shape[:3]

        ## obj_encoding
        objects_features = get_siamese_features(self.object_encoder, obj_points, aggregator=torch.stack) # torch.Size([24, 52, 768])
        obj_feats = self.obj_feature_mapping(objects_features) # torch.Size([24, 52, 768])
        box_infos = self.box_feature_mapping(boxs.float())
        obj_infos = obj_feats[:, None].repeat(1, self.view_number, 1, 1).squeeze() + box_infos # torch.Size([24, 4, 52, 768])
        if len(obj_infos.shape) == 3:
            assert self.view_number == 1
            obj_infos = obj_infos.unsqueeze(1).repeat(1, self.view_number, 1, 1)
        ## language_encoding
        lang_tokens = batch['lang_tokens']
        lang_infos = self.language_encoder(**lang_tokens)[0]

        # <LOSS>: lang_cls
        lang_features = lang_infos[:,0]

        LANG_LOGITS = self.language_clf(lang_infos[:,0])
        if self.lang_multilabel:
            ANCHOR_LOGITS = self.anchor_clf(lang_infos[:,0])
        mem_infos = lang_infos[:, None].repeat(1, self.view_number, 1, 1).reshape(B*self.view_number, -1, self.inner_dim)

        # start feature encoding
        mentioned_obj_lang_infos = self.language_encoder(**batch['order_tokens'])[0]

        mentioned_obj_lang_infos = mentioned_obj_lang_infos.view(B, self.order_len, batch['order_tokens']['input_ids'].size(1), -1) # torch.Size([24, 4, tokenNum, 768])
        
        cat_infos = obj_infos.reshape(B*self.view_number, -1, self.inner_dim) # torch.Size([96, 52, 768])

        # <LOSS>: obj_cls
        if self.label_lang_sup:
            label_lang_infos = self.language_encoder(**self.class_name_tokens)[0][:,0] # torch.Size([525, 768])
            CLASS_LOGITS = torch.matmul(obj_feats.reshape(B*N,-1), label_lang_infos.permute(1,0)).reshape(B,N,-1)
        else:
            CLASS_LOGITS = self.obj_clf(obj_feats.reshape(B*N,-1)).reshape(B,N,-1) # torch.Size([24, 52, 525])        

        for i in range(self.order_len):
            mask = batch['pred_class_mask'][:, i, :].unsqueeze(1).unsqueeze(3).repeat(1, self.view_number, 1, self.inner_dim)
            mask = mask.reshape(B*self.view_number, -1, self.inner_dim)

            masked_obj_infos = cat_infos * mask
            mentioned_features = mentioned_obj_lang_infos[:, i, :, :].unsqueeze(1).repeat(1, self.view_number, 1, 1).reshape(B*self.view_number, -1, self.inner_dim)
            cat_infos = self.refer_encoder[i](
                cat_infos.transpose(0, 1),
                masked_obj_infos.transpose(0, 1),
                mem_infos.transpose(0, 1),
                mentioned_features.transpose(0, 1),
            ) # torch.Size([96, 52, 768])
            if self.multilabel_pretraining:
                if not self.disable_multilabel_loss:
                    TB_MULTILABEL_LOGITS = self.feat_to_multilabel_clf(cat_infos).squeeze() # torch.Size([96, 52])
                    ans = batch['ordered_multilabel_gt'][:, i, :].unsqueeze(1).repeat(1, self.view_number, 1).reshape(B*self.view_number, -1)
                    TB_MULTILABEL_LOSS = self.ml_feature_constraint_loss(TB_MULTILABEL_LOGITS, ans.float())
                    TOTAL_LOSS += TB_MULTILABEL_LOSS * 0.5

        ## multi-modal_fusion
        out_feats = cat_infos.reshape(B, self.view_number, -1, self.inner_dim) # torch.Size([24, 4, 52, 768])
        
        ## view_aggregation
        refer_feat = out_feats
        if self.aggregate_type=='avg':
            agg_feats = (refer_feat / self.view_number).sum(dim=1) # torch.Size([24, 52, 768])
        elif self.aggregate_type=='avgmax':
            agg_feats = (refer_feat / self.view_number).sum(dim=1) + refer_feat.max(dim=1).values
        else:
            agg_feats = refer_feat.max(dim=1).values

        # <LOSS>: ref_cls
        LOGITS = self.object_language_clf(agg_feats).squeeze(-1)
        if self.lang_multilabel:
            BASIC_LOSS = self.compute_basic_loss(batch, CLASS_LOGITS, LANG_LOGITS, LOGITS, ANCHOR_LOGITS)
        else:
            BASIC_LOSS = self.compute_basic_loss(batch, CLASS_LOGITS, LANG_LOGITS, LOGITS)
        TOTAL_LOSS += BASIC_LOSS
        
        return TOTAL_LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS
