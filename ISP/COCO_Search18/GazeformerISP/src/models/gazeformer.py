import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.models import ResNetCOCO
from models.positional_encodings import PositionEmbeddingSine2d
import math
from typing import Optional

eps = 1e-16

class attention_module(nn.Module):
    def __init__(self, feature_size, subject_embedding_size, project_size):
        super(attention_module, self).__init__()
        self.feature_size = feature_size
        self.project_size = project_size

        self.proj_feature = nn.Linear(feature_size, project_size, bias=True)
        self.proj_subject = nn.Linear(subject_embedding_size, project_size, bias=True)
        self.attention = nn.Linear(project_size, 1, bias=True)

        self.init_weights()

    def forward(self, features, attention_map, subject_embedding):
        features = features.permute(1, 2, 0)
        aggr_feature = features.unsqueeze(1).unsqueeze(1) * attention_map.unsqueeze(-2)
        aggr_feature = aggr_feature.mean(-1)

        proj_aggr_feature = self.proj_feature(aggr_feature)
        proj_subject_embedding = self.proj_subject(subject_embedding)
        attention_weights = F.softmax(
            self.attention(torch.tanh(proj_aggr_feature + proj_subject_embedding.unsqueeze(1).unsqueeze(1))), 2)
        attention_weights = attention_weights.squeeze(-1)
        return attention_weights

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

class CrossAttentionPredictor(nn.Module):
    def __init__(self, nhead = 8, dropout=0.4, d_model = 512):
        super(CrossAttentionPredictor, self).__init__()
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self._reset_parameters(self.self_attn)
        self._reset_parameters(self.multihead_attn)

    def _reset_parameters(self, mod):
        for p in mod.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else pos + tensor

    def forward(self, tgt, memory, tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                    querypos_embed: Optional[Tensor] = None,
                    patchpos_embed: Optional[Tensor] = None):
        q = k = v = self.with_pos_embed(tgt, querypos_embed)
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        att = self.multihead_attn(query=self.with_pos_embed(tgt, querypos_embed),
                                   key=patchpos_embed(memory),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[1]
        att_logit = torch.log(att + eps)

        return att_logit

class gazeformer(nn.Module):
    def __init__(self, transformer, spatial_dim, args, subject_num, subject_feature_dim, action_map_num, dropout=0.4, max_len = 7, patch_size  = 16, device = "cuda"):
        super(gazeformer, self).__init__()
        self.args = args
        self.spatial_dim = spatial_dim
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        # subject embeddings
        # self.subject_embed = nn.Embedding(subject_num, subject_feature_dim)
        print(f'load user embedding from {args.user_emb_path}')
        self.subject_embed = torch.load(args.user_emb_path)
        #fixation embeddings
        self.querypos_embed = nn.Embedding(max_len,self.hidden_dim)
        #2D patch positional encoding
        self.patchpos_embed = PositionEmbeddingSine2d(spatial_dim, hidden_dim=self.hidden_dim, normalize=True, device = device)
        #2D pixel positional encoding for initial fixation
        self.queryfix_embed = PositionEmbeddingSine2d((spatial_dim[0] * patch_size, spatial_dim[1] * patch_size),
                                                      hidden_dim=self.hidden_dim, normalize=True, flatten = False, device = device).pos
        #classify fixation, or PAD tokens
        self.token_predictor = nn.Linear(self.hidden_dim, action_map_num)
        #Gaussian parameters for x,y,t
        self.generator_t_mu = nn.Linear(self.hidden_dim, 1)
        self.generator_t_logvar = nn.Linear(self.hidden_dim, 1)

        self.max_len = max_len
        
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.LogSoftmax(dim=-1)
        #projection for first fixation encoding
        self.attention_map_predictors = nn.ModuleList([CrossAttentionPredictor(self.args.nhead, dropout, self.args.hidden_dim) for _ in range(action_map_num)])

        # attention modules for the output
        self.attention_module = attention_module(self.hidden_dim, subject_feature_dim, self.hidden_dim // 2)

        self.init_weights()

    def init_weights(self):
        # for modules in [self.subject_embed.modules(), self.querypos_embed.modules(), self.token_predictor.modules(),
        for modules in [self.querypos_embed.modules(), self.token_predictor.modules(),
                        self.generator_t_mu.modules(), self.generator_t_logvar.modules()]:
            for m in modules:
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.01)

    def get_fixation_map(self, y_mu, y_logvar, x_mu, x_logvar):
        y_grid = self.y_grid.unsqueeze(0).unsqueeze(0)
        x_grid = self.x_grid.unsqueeze(0).unsqueeze(0)
        y_mu, y_logvar, x_mu, x_logvar = y_mu.unsqueeze(-1), y_logvar.unsqueeze(-1), x_mu.unsqueeze(-1), x_logvar.unsqueeze(-1)
        y_std = torch.exp(0.5 * y_logvar)
        x_std = torch.exp(0.5 * x_logvar)

        exp_term = (y_grid - y_mu) ** 2 / (y_std ** 2 + eps) + (x_grid - x_mu) ** 2 / (x_std ** 2 + eps)
        fixation_map = 1 / (2 * math.pi) / (y_std + eps) / (x_std + eps) * torch.exp(-0.5 * exp_term)
        fixation_map = fixation_map.view(fixation_map.shape[0], fixation_map.shape[1], -1)
        fixation_map = fixation_map / (fixation_map.sum(-1, keepdim=True) + eps)
        return fixation_map
        
    #reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, src: Tensor, subjects: Tensor, task: Tensor):
        self.subject_embed = self.subject_embed.to(subjects.device)
        if self.training:
            predicts = self.training_process(src, subjects, task)
        else:
            predicts = self.inference(src, subjects, task)

        return predicts

    def training_process(self, src, subjects, task):
        tgt_input = src.new_zeros((self.max_len, src.size(0), self.hidden_dim)) #Notice that this where we convert target input to zeros
        
        # tgt_input[0, :, :] = self.firstfix_linear(self.queryfix_embed[tgt[:, 0], tgt[:,1], :])
        # memory_task, outs = self.transformer(src=src, tgt=tgt_input, tgt_mask= None, tgt_key_padding_mask = None, subjects = self.subject_embed(subjects), task = task,
        #                         querypos_embed = self.querypos_embed.weight.unsqueeze(1), patchpos_embed = self.patchpos_embed)
        memory_task, outs = self.transformer(src=src, tgt=tgt_input, tgt_mask= None, tgt_key_padding_mask = None, subjects = self.subject_embed[subjects], task = task,
                                querypos_embed = self.querypos_embed.weight.unsqueeze(1), patchpos_embed = self.patchpos_embed)

        outs = self.dropout(outs)
        #get Gaussian parameters for (t)
        t_log_normal_mu, t_log_normal_sigma2 = self.generator_t_mu(outs), torch.exp(self.generator_t_logvar(outs))
        action_maps = []
        for attention_map_predictor in self.attention_map_predictors:
            action_map = attention_map_predictor(outs, memory_task, tgt_mask=None, memory_mask=None,
                                                 tgt_key_padding_mask=None, memory_key_padding_mask=None,
                                                 querypos_embed = self.querypos_embed.weight.unsqueeze(1),
                                                 patchpos_embed=self.patchpos_embed)
            action_maps.append(action_map)
        action_maps = torch.stack(action_maps, dim=-2)
        # get the attention weight
        attention_weights = self.attention_module(memory_task, action_maps, self.subject_embed[subjects])

        token_prediction = self.token_predictor(outs).permute(1, 0, 2)
        z = torch.cat([token_prediction.unsqueeze(-1), action_maps], dim=-1)

        aggr_action_map = (action_maps * attention_weights.unsqueeze(-1)).sum(2)
        aggr_z = (z * attention_weights.unsqueeze(-1)).sum(2)

        if self.training == False:
            aggr_z = F.softmax(aggr_z, -1)


        predicts = {}
        # [N, T, A] A = H * W + 1
        predicts['actions'] = aggr_z
        # [N, T]
        predicts['log_normal_mu'] = t_log_normal_mu.permute(1, 0, 2).squeeze(2)
        # [N, T]
        predicts['log_normal_sigma2'] = t_log_normal_sigma2.permute(1, 0, 2).squeeze(2)
        # [N, T, H, W]
        predicts["action_map"] = aggr_action_map.view(-1, self.max_len, self.spatial_dim[0], self.spatial_dim[1])

        return predicts
        
    def inference(self, src, subjects, task):
        tgt_input = src.new_zeros((self.max_len, src.size(0), self.hidden_dim)) #Notice that this where we convert target input to zeros
        
        # tgt_input[0, :, :] = self.firstfix_linear(self.queryfix_embed[tgt[:, 0], tgt[:,1], :])
        memory_task, outs = self.transformer(src=src, tgt=tgt_input, tgt_mask= None, tgt_key_padding_mask = None, subjects = self.subject_embed[subjects], task = task,
                                querypos_embed = self.querypos_embed.weight.unsqueeze(1), patchpos_embed = self.patchpos_embed)

        outs = self.dropout(outs)
        #get Gaussian parameters for (t)
        t_log_normal_mu, t_log_normal_sigma2 = self.generator_t_mu(outs), torch.exp(self.generator_t_logvar(outs))
        action_maps = []
        for attention_map_predictor in self.attention_map_predictors:
            action_map = attention_map_predictor(outs, memory_task, tgt_mask=None, memory_mask=None,
                                                 tgt_key_padding_mask=None, memory_key_padding_mask=None,
                                                 querypos_embed = self.querypos_embed.weight.unsqueeze(1),
                                                 patchpos_embed=self.patchpos_embed)
            action_maps.append(action_map)
        action_maps = torch.stack(action_maps, dim=-2)
        # get the attention weight
        attention_weights = self.attention_module(memory_task, action_maps, self.subject_embed[subjects])

        token_prediction = self.token_predictor(outs).permute(1, 0, 2)
        z = torch.cat([token_prediction.unsqueeze(-1), action_maps], dim=-1)

        aggr_action_map = (action_maps * attention_weights.unsqueeze(-1)).sum(2)
        aggr_z = (z * attention_weights.unsqueeze(-1)).sum(2)

        if self.training == False:
            aggr_z = F.softmax(aggr_z, -1)


        predicts = {}
        # [N, T, A] A = H * W + 1
        predicts['all_actions_prob'] = aggr_z
        # [N, T]
        predicts['log_normal_mu'] = t_log_normal_mu.permute(1, 0, 2).squeeze(2)
        # [N, T]
        predicts['log_normal_sigma2'] = t_log_normal_sigma2.permute(1, 0, 2).squeeze(2)
        # [N, T, H, W]
        predicts["action_map"] = aggr_action_map.view(-1, self.max_len, self.spatial_dim[0], self.spatial_dim[1])
        
        return predicts