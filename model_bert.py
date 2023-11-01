# -----------------------------------------------------------
# Dual Semantic Relations Attention Network (DSRAN) implementation based on
# "VSE++: Improving Visual-Semantic Embeddings with Hard Negatives"
# "Learning Dual Semantic Relations with Graph Attention for Image-Text Matching"
# Keyu Wen, Xiaodong Gu, and Qingrong Cheng
# IEEE Transactions on Circuits and Systems for Video Technology, 2020
# Writen by Keyu Wen & Linyang Li, 2020
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import copy
from resnet import resnet152
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
import time
from GAT import GATLayer
from GAT_1 import IGSAN
from adapt import ADAPT
# from GAT_2 import IGSAN1
from relative_embedding import BoxRelationalEmbedding


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B*N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x

class RcnnEncoder(nn.Module):
    def __init__(self, opt):
        super(RcnnEncoder, self).__init__()
        self.embed_size = opt.embed_size
        self.fc_image = nn.Linear(opt.img_dim, self.embed_size)
        self.mlp = MLP(opt.img_dim, opt.embed_size // 2, opt.embed_size, 2)

    def forward(self, images):  # (b, 100, 2048) (b,100,1601+6)
        img_f = self.fc_image(images)
        img_f = self.mlp(images) + img_f
        img_f = l2norm(img_f)
        # img_pe = self.fc_pos(img_pos)
        # img_embs = img_f + img_pe
        return img_f # (b,100,768)

class GridEncoder(nn.Module):
    def __init__(self, opt):
        super(GridEncoder, self).__init__()
        self.embed_size = opt.embed_size
        self.fc_grid = nn.Linear(opt.img_dim, self.embed_size)
        self.init_weights()
        
    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc_grid.in_features +
                                  self.fc_grid.out_features)
        self.fc_grid.weight.data.uniform_(-r, r)
        self.fc_grid.bias.data.fill_(0)


    def forward(self, img_grid):  # (b, 100, 2048) (b,100,1601+6)
        img_g = self.fc_grid(img_grid)
        img_g = l2norm(img_g)
        return img_g





class TextEncoder(nn.Module):
    def __init__(self, opt):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(opt.bert_path)
        if not opt.ft_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print('text-encoder-bert no grad')
        else:
            print('text-encoder-bert fine-tuning !')
        self.embed_size = opt.embed_size
        self.fc = nn.Linear(opt.bert_size, opt.embed_size)

    def forward(self, captions):
        all_encoders, pooled = self.bert(captions)
        out = all_encoders[-1]
        out = self.fc(out)
        return out

def func_attention_i2t(query, context, smooth, eps=1e-8, weight=None):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)


        # --> (batch*sourceL, queryL)
    attn = attn.view(batch_size * sourceL, queryL)
    attn = F.softmax(attn, dim=1)
    # --> (batch, sourceL, queryL)
    attn = attn.view(batch_size, sourceL, queryL)

    if weight is not None:
        attn = attn + weight

    attn_out = attn.clone()

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)

    attn = F.softmax(attn * smooth, dim=1)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attn_out

class GATopt(object):
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = 8
        self.hidden_dropout_prob = 0.2
        self.attention_probs_dropout_prob = 0.2


class GAT(nn.Module):
    def __init__(self, config_gat):
        super(GAT, self).__init__()
        layer = GATLayer(config_gat)
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(config_gat.num_layers)])

    def forward(self, querys, keys, values, attention_mask=None, position_weight = None):
        hidden_states = querys
        for layer_module in self.encoder:
            hidden_states = layer_module(querys, keys, values, attention_mask, position_weight)
        return hidden_states  # B, seq_len, D


def cosine_sim(im, s):
    return im.mm(s.t())

def pdist_cos(x1, x2):
    """
        compute cosine similarity between two tensors
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise cosine distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1_norm = x1 / x1.norm(dim=1)[:, None]
    x2_norm = x2 / x2.norm(dim=1)[:, None]
    res = torch.mm(x1_norm, x2_norm.transpose(0, 1))
    mask = torch.isnan(res)
    res[mask] = 0
    return res

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        # self.sim = cosine_sim

    def forward(self, scores):
        # scores = self.sim(im, s)
        diagonal = scores.diag().view(scores.size(0), 1)

        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)

        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


def get_optimizer(params, opt, t_total=-1):
    bertadam = BertAdam(params, lr=opt.learning_rate, warmup=opt.warmup, t_total=t_total)
    return bertadam


class Fusion(nn.Module):
    def __init__(self, opt):
        super(Fusion, self).__init__()
        self.f_size = opt.embed_size
        # self.gate0 = nn.Linear(self.f_size, self.f_size)
        # self.gate1 = nn.Linear(self.f_size, self.f_size)

        self.fusion1 = nn.Linear(self.f_size * 2, self.f_size)
        self.relu = nn.ReLU()
        self.fusion2 = nn.Linear(self.f_size, self.f_size)
        self.bn = nn.BatchNorm1d(self.f_size)
        # self.fusion3 = nn.Linear(self.f_size * 2, self.f_size)
        # self.sigmoid = torch.sigmoid()

    def forward(self, vec1, vec2):
        vec_cat = torch.cat([vec1, vec2], dim=-1)
        map1_out = self.fusion1(vec_cat)
        map1_out = self.relu(map1_out)
        map2_out = self.fusion2(map1_out)
        gate = torch.sigmoid(torch.mul(map2_out, vec2))
        gate_result = gate * vec2
        # concat_result = torch.cat([vec1, gate_result], dim=-1)
        # concat_result = self.relu(self.fusion3(concat_result))
        gate_result = self.bn(gate_result.permute(0, 2, 1)).permute(0, 2, 1)
        f = gate_result + vec1
        return f

class weightpool(nn.Module):
    def __init__(self, opt):
        super(weightpool, self).__init__()
        self.fc1 = nn.Linear(opt.embed_size, opt.embed_size)
        self.fc2 = nn.Linear(opt.embed_size, opt.embed_size)
        self.act = nn.ReLU()
    
    def forward(self, vec):
        out_features = self.fc1(vec)
        out_features = self.act(out_features)
        out_features = self.fc2(out_features)
        out_weights = nn.Softmax(dim=1)(out_features)
        out_emb = torch.mul(vec, out_weights)
        out_emb = out_emb.permute(0, 2, 1)
        pool_emb = torch.sum(out_emb.view(out_emb.size(0), out_emb.size(1), -1), dim=2)
        return pool_emb




class DSRAN(nn.Module):
    def __init__(self, opt):
        super(DSRAN, self).__init__()
        self.K = opt.K
        # self.img_enc = EncoderImageFull(opt)
        self.rcnn_enc = RcnnEncoder(opt)
        self.grid_enc = GridEncoder(opt)
        self.txt_enc = TextEncoder(opt)
        # config_rcnn = GATopt(opt.embed_size, 1)
        # config_img= GATopt(opt.embed_size, 1)
        # region_grid = GATopt(opt.embed_size, 1)
        # grid_region = GATopt(opt.embed_size, 1)
        config_cap= GATopt(opt.embed_size, 1)
        config_joint= GATopt(opt.embed_size, 1)
        # SSR
        # self.gat_1 = GAT(config_rcnn)
        # self.gat_2 = GAT(config_img)
        self.gat_cat = GAT(config_joint)
        # JSR
        self.region_intra = IGSAN(1, opt.embed_size, 8, is_share = False, drop = 0.2)
        self.grid_intra = IGSAN(1, opt.embed_size, 8, is_share = False, drop = 0.2)

        self.gat_cat_1 = IGSAN(1, opt.embed_size, 8, is_share = False, drop = 0.2)
        self.gat_cat_2 = IGSAN(1, opt.embed_size, 8, is_share = False, drop = 0.2)
        self.fusion1 = Fusion(opt)
        self.fusion2 = Fusion(opt)
        self.gat_cap = GAT(config_cap)
        self.region_pool = weightpool(opt)
        self.grid_pool = weightpool(opt)
        self.cap_pool = weightpool(opt)
        # self.cap_pool1 = weightpool(opt)
        # self.fc1 = nn.Linear(opt.embed_size, opt.embed_size)
        # self.fc2 = nn.Linear(opt.embed_size, opt.embed_size)
        # self.fusion4 = nn.Linear(opt.embed_size, opt.embed_size)
        # self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(8)])
        
    def forward(self, img_rcnn, img_grid, img_mask, region_mask, grid_mask, captions):

        n_regions = img_rcnn.shape[1]
        n_grids = img_grid.shape[1]
        bs = img_rcnn.shape[0]

        rcnn_emb_raw = self.rcnn_enc(img_rcnn)
        rcnn_grid = self.grid_enc(img_grid)
        out_region = rcnn_emb_raw
        out_grid = rcnn_grid
        # out_region = self.gat_1(rcnn_emb, rcnn_emb, rcnn_emb)
        # out_grid = self.gat_2(rcnn_grid, rcnn_grid, rcnn_grid)
        attention_mask = img_mask.unsqueeze(1)
        # print(attention_mask.shape)
        region_mask = region_mask.unsqueeze(1)
        # print(region_mask.shape)
        # grid_mask = grid_mask.unsqueeze(1)
        # print(grid_mask.shape)
        # region_aligns = (torch.cat([region_mask, attention_mask], dim=-1) == 0)
        # grid_aligns1 = (torch.cat([attention_mask.permute(0, 1, 3, 2), grid_mask], dim=-1)==0)
        # print(grid_aligns1[0][0][0])
        # print(attention_mask.shape)

        tmp_mask = torch.eye(n_regions, n_regions, device=out_region.device).unsqueeze(0).unsqueeze(0)
        # print(tmp_mask.shape)
        tmp_mask = tmp_mask.repeat(bs, 1, 1, 1)  # bs * 1 * n_regions * n_regions
        # print(tmp_mask.shape)
        region_aligns = (torch.cat([tmp_mask, attention_mask], dim=-1) == 0) # bs * 1 * n_regions *(n_regions+n_grids)
        # print(region_aligns[0][0][0])

        tmp_mask = torch.eye(n_grids, n_grids, device=out_grid.device).unsqueeze(0).unsqueeze(0)
        tmp_mask = tmp_mask.repeat(bs, 1, 1, 1)  # bs * 1 * n_grids * n_grids
        grid_aligns = (torch.cat([attention_mask.permute(0, 1, 3, 2), tmp_mask], dim=-1)==0) # bs * 1 * n_grids *(n_grids+n_regions)   
        # print(grid_aligns[0][0][0])

        region_alls = torch.cat([region_aligns, grid_aligns], dim=-2)
        # print(region_alls.shape)

        rcnn_emb = self.region_intra(rcnn_emb_raw, rcnn_emb_raw, rcnn_emb_raw, region_mask)
        rcnn_grid = self.grid_intra(rcnn_grid, rcnn_grid, rcnn_grid)

        out_all = torch.cat([rcnn_emb, rcnn_grid], 1)
        out1 = self.gat_cat(out_all, out_all, out_all, region_alls)

        region_self = out1[:,:36,:]
        grid_self = out1[:,36:85,:]

        region_other = self.gat_cat_1(region_self, grid_self, grid_self)
        grid_other = self.gat_cat_2(grid_self, region_self, region_self)

        region_grid = self.fusion1(region_self, region_other)
        grid_region = self.fusion2(grid_self, grid_other)
        
        out_region_emb = self.region_pool(region_grid)
        out_grid_emb = self.grid_pool(grid_region)

        img_cat = out_region_emb + out_grid_emb
        # img_embs = l2norm(img_cat)
        # img_gate = torch.sigmoid(self.fusion3(out_region_emb) + self.fusion4(out_grid_emb))


        cap_emb = self.txt_enc(captions)
        cap_gat = self.gat_cap(cap_emb, cap_emb, cap_emb)
        out_cap_emb = self.cap_pool(cap_gat)
        # out_cap_emb = torch.mean(out_cap_emb, 1)
        # cap_embs = l2norm(out_cap_emb)

        # img_code_loc1 = self.fc1(rcnn_emb)
        # cap_img, _ = func_attention_i2t(img_code_loc1, cap_emb, smooth=4) #(batch_size, worn_num, embed_size)
        # cap_img = self.fc2(cap_img)
        # text_img = self.cap_pool1(cap_img)

        # cap_final_emb = out_cap_emb + text_img
        # cap_embs = l2norm(out_cap_emb)

        # all_cat = out_region_emb + out_grid_emb + cap_embs
        # gate = torch.sigmoid(self.fc(all_cat))
        # # print(gate.shape)
        # img_cat = gate * out_region_emb + (1-gate) * out_grid_emb
        # img_embs = l2norm(img_cat)

        return img_cat, out_cap_emb, rcnn_emb_raw, cap_emb




def func_attention(query, context, raw_feature_norm, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)   #(n, d, qL)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)   #(n, cL, qL)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = F.softmax(attn, dim=-1)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        # attn = l2norm(attn, 2)
        attn = F.normalize(attn, dim=2)
    elif raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous() #(n, qL, cL)
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)    #(n*qL, cL)
    attn = F.softmax(attn*smooth, dim=-1)                #(n*qL, cL)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)   #(n, qL, cL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()    #(n, cL, qL)

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)   #(n, d, cL)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)    #(n, d, qL)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)    #(n, qL, d)

    return weightedContext, attnT

class Fovea(nn.Module):

    def __init__(self, smooth=10, train_smooth=False):
        super().__init__()

        self.smooth = smooth
        self.train_smooth = train_smooth
        self.softmax = nn.Softmax(dim=-1)

        if train_smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + self.smooth)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        mask = self.softmax(x * self.smooth)
        output = mask * x
        return output

    def __repr__(self):
        return (
            f'Fovea(smooth={self.smooth},'
            f'train_smooth: {self.train_smooth})'
        )

class AdaptiveEmbeddingI2T(nn.Module):
    def __init__(self, opt, q1_size=1024, q2_size=1024, v1_size=1024, v2_size=1024, k=1):
        super(AdaptiveEmbeddingI2T, self).__init__()
        latent_size = opt.embed_size
        self.norm = nn.BatchNorm1d(latent_size, affine=False)
        self.adapt_txt = ADAPT(k, v1_size, v2_size, q1_size, q2_size, nonlinear_proj = True)
        self.cap_pool = weightpool(opt)
        # self.fovea = Fovea(smooth=10, train_smooth=False)

    def forward(self, img_glo, cap_glo, img_embed, cap_embed):
        '''
            img_embed: (B, 36, latent_size)
            cap_embed: (B, T, latent_size)
        '''
        img_embed = img_embed.permute(0, 2, 1)
        cap_embed = l2norm(cap_embed)
        cap_embed = cap_embed.permute(0, 2, 1)
        img_embed = self.norm(img_embed)
    
        sims = torch.zeros(img_embed.shape[0], cap_embed.shape[0]).to(device='cuda')

        for i, cap_tensor in enumerate(cap_embed):
            cap_repr = cap_tensor[:, :32].mean(-1).unsqueeze(0)
            cap_repr1 = cap_glo[i].unsqueeze(0)
            # cap_repr = cap_tensor.unsqueeze(0)
            q2_repr = None

            img_output = self.adapt_txt(img_embed, img_glo, cap_repr, q2_repr)
            # print(img_output.shape)
            img_output = img_output.permute(0, 2, 1)
            img_cap = self.cap_pool(img_output)
            img_fin = img_glo + img_cap
            # img_vector = self.cap_pool(img_fin)

            # img_output = self.fovea(img_output)
            # img_vector = img_output.mean(-1)
            
            img_fin = l2norm(img_fin)
            cap_vector = l2norm(cap_repr1)
            sim = cosine_sim(img_fin, cap_vector).squeeze(-1)
            # sim = row_sim.squeeze(-1)
            sims[:, i] = sim
        #     # print(row_sim.shape)
        # sims = torch.cat(sims, 0)

        # for i in range(n_caption):
        #     n_word = 32
        #     cap_i = cap_embed[i, :n_word, :].unsqueeze(0).contiguous()
        #     cap_i = cap_i.permute(0, 2, 1)
        #     cap_glo_i = cap_glo[i, :].unsqueeze(0).contiguous()
        #     # print(cap_glo_i.shape)
        #     cap_i_expand = cap_i.repeat(n_image, 1, 1)
        #     cap_glo_i_expand = cap_glo_i.repeat(n_image, 1)
        #     weiContext = torch.mean(img_embed, 1)
        #     ref_wrd = self.adapt_txt(value=cap_i_expand, query=weiContext)
        #     ref_wrd = ref_wrd.permute(0, 2, 1)
        #     # weiContext, attn = func_attention(cap_i_expand, img_embed, self.raw_feature_norm, smooth=self.lambda_softmax)
        #     # ref_wrd = self.refine(cap_i_expand, weiContext)
        #     txt_output = self.cap_pool1(ref_wrd)
        #     # txt_output = torch.mean(ref_wrd, 1)
        #     txt_glo = cap_glo_i_expand + txt_output
        #     txt_glo = l2norm(txt_glo, 1)

        #     row_sim = cosine_sim(img_glo, txt_glo)
        #     # print(row_sim.shape)
        #     row_sim = row_sim.max(dim=1, keepdim=True)[0]
        #     # print(row_sim.shape)
        #     sims.append(row_sim)
        # sims = torch.cat(sims, 1)

        return sims



class VSE(object):

    def __init__(self, opt):
        self.DSRAN = DSRAN(opt)
        self.sim_enc = AdaptiveEmbeddingI2T(opt)
        self.DSRAN = nn.DataParallel(self.DSRAN)
        self.sim_enc = nn.DataParallel(self.sim_enc)
        if torch.cuda.is_available():
            self.DSRAN.cuda()
            self.sim_enc.cuda()
            cudnn.benchmark = True
        self.criterion = ContrastiveLoss(margin=opt.margin)
        params = list(self.DSRAN.named_parameters())
        params += list(self.sim_enc.named_parameters())
        param_optimizer = params
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = opt.l_train * opt.num_epochs
        if opt.warmup == -1:
            t_total = -1
        self.optimizer = get_optimizer(params=optimizer_grouped_parameters, opt=opt, t_total=t_total)
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.DSRAN.state_dict(), self.sim_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.DSRAN.load_state_dict(state_dict[0])
        self.sim_enc.load_state_dict(state_dict[1])

    def train_start(self):
        self.DSRAN.train()
        self.sim_enc.train()

    def val_start(self):
        self.DSRAN.eval()
        self.sim_enc.eval()

    def forward_emb(self, img_rcnn, img_grid, img_mask, region_mask, grid_mask, captions):
        if torch.cuda.is_available():
            img_rcnn = img_rcnn.cuda()
            img_grid = img_grid.cuda()
            img_mask = img_mask.cuda()
            region_mask = region_mask.cuda()
            grid_mask = grid_mask.cuda()
            # img_pos = img_pos.cuda()
            # img_box = img_box.cuda()
            captions = captions.cuda()

        img_glo, cap_glo, img_emb, cap_emb = self.DSRAN(img_rcnn, img_grid, img_mask, region_mask, grid_mask,captions)

        return img_glo, cap_glo, img_emb, cap_emb

    def forward_sim(self, img_glo, cap_glo, img_emb, cap_emb):
        # Forward similarity encoding
        sims = self.sim_enc(img_glo, cap_glo, img_emb, cap_emb)
        return sims

    def forward_loss(self, sims, **kwargs):
        loss = self.criterion(sims)
        self.logger.update('Le', loss.data, sims.size(0))
        return loss

    def train_emb(self, img_rcnn, img_grid, img_mask, region_mask, grid_mask, captions, ids=None, *args):
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        img_glo, cap_glo, img_emb, cap_emb = self.forward_emb(img_rcnn, img_grid, img_mask, region_mask, grid_mask, captions)
        sims = self.forward_sim(img_glo, cap_glo, img_emb, cap_emb)

        self.optimizer.zero_grad()
        loss = self.forward_loss(sims)

        loss.backward()
        self.optimizer.step()
