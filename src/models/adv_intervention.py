# src/models/adv_intervention.py
import torch
from pyvene import SourcelessIntervention, DistributedRepresentationIntervention
import pdb
class AdversarialIntervention(
    SourcelessIntervention,
    DistributedRepresentationIntervention
):
    """
    一个“假的” ReFT 插件：只是在对应层对 base hidden 加上 delta。
    delta 不作为模型参数保存，只在 inner-loop PGD 里手动设置。
    """
    def __init__(self, **kwargs):
        # embed_dim 会由 ReftConfig 传进来
        super().__init__(**kwargs, keep_last_dim=True)
        self.delta = None   
        self.last_base = None # 记录最后一次 forward 的输入
        self.last_out = None # 记录最后一次 forward 的输出
    def reset_delta(self):
        """
        PGD 每一步由外部调用：adv_intervention.set_delta(delta)
        传 None 表示当前步不加扰动
        """
        self.delta = None

    def forward(self, base, source=None, subspaces=None):
        """
        base: [B, S_sel, D] 这里 S_sel 可能是 1（只最后一个 token），不一定是 512。
        """
        self.last_base = base.detach()
        # 第一次或者被 reset 后：按当前 base 形状建一个 0 的 delta  
        if self.delta is None:  
            self.delta = torch.zeros_like(base, requires_grad=True)  
        # 移除形状检查，直接使用现有的 delta  
        else:  
            # 确保仍然是 graph 里的 leaf  
            if not self.delta.requires_grad:  
                self.delta.requires_grad_(True)  
        out = base + self.delta.to(base.dtype).to(base.device)
        self.last_out = out.detach()
        
        # ###################### [debug]
        # if getattr(self, "_dbg_count", 0) < 5:
        #     self._dbg_count = getattr(self, "_dbg_count", 0) + 1
        #     with torch.no_grad():
        #         bn = base.float().norm(dim=-1).mean().item()
        #         print(f"[ADV_FWD] base shape={tuple(base.shape)} base_norm_mean={bn:.4f} delta_shape={None if self.delta is None else tuple(self.delta.shape)}")
        #         if self.delta is not None:
        #             dn = self.delta.detach().float().norm(dim=-1).mean().item()
        #             dmax = self.delta.detach().abs().max().item()
        #             print(f"[ADV_FWD] delta_norm_mean={dn:.4f} delta_absmax={dmax:.4f}")
        # #####################
        return out