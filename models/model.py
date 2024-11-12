import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial, reduce
from models.modules import ResNet101, ASPP


class DeepLabV3(nn.Module):
    def __init__(
        self,
        output_stride=16,
        norm_act='bn_sync',
        backbone_pretrained=False,
        classes=None,
        method=None,
        use_cosine=False,
        freeze_all_bn=False,
        freeze_backbone_bn=False,
    ):
        super().__init__()
        self.norm_act = norm_act
        self.method = method
        if norm_act == 'iabn_sync':
            from inplace_abn import ABN, InPlaceABNSync
            norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01)
        elif norm_act == 'bn_sync':
            norm = nn.BatchNorm2d
        else:
            raise NotImplementedError

        self.freeze_all_bn = freeze_all_bn
        self.freeze_backbone_bn = freeze_backbone_bn

        self.classes = classes
        self.tot_classes = reduce(lambda a, b: a + b, self.classes) + 1 if self.method == 'DKD' else reduce(lambda a, b: a + b, self.classes)
        self.use_cosine = use_cosine
        
        self.aspp_last_dim = 256
        
        use_bias = not use_cosine

        # Network
        self.backbone = ResNet101(norm, norm_act, output_stride, backbone_pretrained)
        self.aspp = ASPP(2048, self.aspp_last_dim, self.aspp_last_dim, norm_act=norm_act, norm=norm, output_stride=output_stride)
        
        if self.method == 'DKD':
            self.cls = nn.ModuleList([nn.Conv2d(self.aspp_last_dim, c, kernel_size=1, bias=use_bias) for c in [1] + classes])  # cls[0]: an auxiliary classifier
            self._init_classifier()
        elif self.method == 'MiB' or self.method == 'PLOP':
            self.cls = nn.ModuleList([nn.Conv2d(self.aspp_last_dim, c, kernel_size=1, bias=use_bias) for c in classes])
            self._init_classifier()
        else:
            raise ValueError("Not implemented Yet")

    def forward(self, x, ret_intermediate=False):
        B, C, H, W = x.size()  # spatial size
        
        _, x_pl, attentions = self.forward_before_class_prediction(x)

        sem_logits_small = self.forward_class_prediction(x_pl)
        sem_logits = F.interpolate(
            sem_logits_small, size=[H, W],
            mode="bilinear", align_corners=False
        )
            
        if ret_intermediate:
            attentions_list = attentions+[x_pl]
            
            if self.method == 'DKD':
                sem_neg_logits_small = self.forward_class_prediction_negative(x_pl)
                sem_pos_logits_small = self.forward_class_prediction_positive(x_pl)
                
                return sem_logits, {'neg_reg': sem_neg_logits_small, 'pos_reg': sem_pos_logits_small, 'attentions': attentions_list}
            elif self.method == 'PLOP':
                return sem_logits, {'attentions': attentions_list, 'sem_logits_small': sem_logits_small }
            elif self.method == "MiB":
                return sem_logits, {'attentions': attentions_list}
            else:
                raise ValueError("Not Implemented Yet")
        else:
            return sem_logits, {}
        
    def forward_before_class_prediction(self, x):
        x_b, attentions = self.backbone(x)
        x_pl = self.aspp(x_b)
        return x_b, x_pl, attentions

    def forward_class_prediction(self, x_pl):
        out = []
        for i, mod in enumerate(self.cls):
            if i == 0 and self.method=='DKD':
                out.append(mod(x_pl.detach()))  # [N, c, H, W]
            else:
                out.append(mod(x_pl))  # [N, c, H, W]
        x_o = torch.cat(out, dim=1)  # [N, |Ct|, H, W]
        return x_o

    def forward_class_prediction_negative(self, x_pl):
        # x_pl: [N, C, H, W]
        out = []
        for i, mod in enumerate(self.cls):
            if i == 0:
                continue
            w = mod.weight  # [|C|, c]
            w = w.where(w < 0, torch.zeros_like(w, device=w.device))
            out.append(torch.matmul(x_pl.permute(0, 2, 3, 1), w.T).permute(0, 3, 1, 2))  # [N, |C|, H, W]
        x_o = torch.cat(out, dim=1)  # [N, |Ct|, H, W]
        return x_o

    def forward_class_prediction_positive(self, x_pl):
        # x_pl: [N, C, H, W]
        out = []
        for i, mod in enumerate(self.cls):
            if i == 0:
                continue
            w = mod.weight  # [|C|, c]
            w = w.where(w > 0, torch.zeros_like(w, device=w.device))
            out.append(torch.matmul(x_pl.permute(0, 2, 3, 1), w.T).permute(0, 3, 1, 2))  # [N, |C|, H, W]
        x_o = torch.cat(out, dim=1)  # [N, |Ct|, H, W]
        return x_o            

    def _init_classifier(self):
        # Random Initialization
        for m in self.cls.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def init_novel_classifier_DKD(self):
        # Initialize novel classifiers using an auxiliary classifier
        cls = self.cls[-1]  # New class classifier
        for i in range(self.classes[-1]):
            cls.weight[i:i + 1].data.copy_(self.cls[0].weight)
            cls.bias[i:i + 1].data.copy_(self.cls[0].bias)
            
    def init_novel_classifier_MiB(self):
        cls = self.cls[-1]
        
        imprinting_w = self.cls[0].weight[0]
        if not self.use_cosine:
            bkg_bias = self.cls[0].bias[0]
            
        if not self.use_cosine:
            # [self.classes[-1] + 1] : new classes + bg class
            bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(bkg_bias.device)
            new_bias = (bkg_bias - bias_diff)
        
        cls.weight.data.copy_(imprinting_w)
        if not self.use_cosine:
            cls.bias.data.copy_(new_bias)
        
        if not self.use_cosine:
            self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def freeze_bn(self, affine_freeze=False):
        if self.freeze_all_bn:
            for m in self.modules():
                if self.norm_act == 'bn_sync':
                    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        m.eval()
                        if affine_freeze is True:
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
                elif self.norm_act == 'iabn_sync':
                    from inplace_abn import ABN, InPlaceABNSync
                    if isinstance(m, (ABN)):
                        m.eval()
                        if affine_freeze is True:
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
            
        elif self.freeze_backbone_bn:
            for m in self.backbone.modules():
                if self.norm_act == 'bn_sync':
                    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        m.eval()
                        if affine_freeze is True:
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
                elif self.norm_act == 'iabn_sync':
                    from inplace_abn import ABN, InPlaceABNSync
                    if isinstance(m, (ABN)):
                        m.eval()
                        if affine_freeze is True:
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False

    def freeze_dropout(self):
        for m in self.modules():
            if isinstance(m, (nn.Dropout)):
                m.eval()

    def _load_pretrained_model(self, pretrained_path):
        pretrain_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
        self.load_state_dict(pretrain_dict['state_dict'], strict=False)

    def _set_bn_momentum(self, model=None, momentum=0.1):
        if model is not None:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.momentum = momentum
        else:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.momentum = momentum

    def get_backbone_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
                elif isinstance(m[1], (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    if not (self.freeze_all_bn or self.freeze_backbone_bn):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_aspp_params(self):
        modules = [self.aspp]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
                elif isinstance(m[1], (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    if not (self.freeze_all_bn):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
        
    def get_classifier_params(self):
        modules = [self.cls]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_old_classifier_params(self):
        modules = [self.cls[i] for i in range(0, len(self.cls) - 1)]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_new_classifier_params(self):
        modules = [self.cls[len(self.cls) - 1]]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], (nn.Conv2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
