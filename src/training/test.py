# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-

# # """
# # inspect_reft_model.py
# # 一键打印：
# #   1) reft_model 的模块树 (named_modules)
# #   2) 参数名、形状、dtype、device、requires_grad (named_parameters)
# #   3) 针对 LoReFT 模块，额外打印 rotate_layer.weight 与 parametrizations.weight[0].base 的区别
# #   4) 做一次最小前向 (batch_size=1) 并打印 reft_model(base=...) 的返回键，方便你确认 out 的结构
# # """

# # import argparse
# # from pathlib import Path
# # import torch
# # from transformers import AutoTokenizer, AutoModelForSequenceClassification
# # import pyreft
# # from pyreft import ReftConfig

# # def header(msg):
# #     print("\n" + "=" * 80)
# #     print(msg)
# #     print("=" * 80)

# # def find_loreft_modules(reft_model):
# #     out = []
# #     for name, mod in reft_model.named_modules():
# #         if isinstance(mod, pyreft.LoreftIntervention):
# #             out.append((name, mod))
# #     return out

# # def build_model_and_reft(args):
# #     device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

# #     # 1) load base model & tokenizer
# #     model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
# #     tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
# #     if tok.pad_token is None:
# #         tok.pad_token = tok.eos_token
# #     model.config.pad_token_id = tok.pad_token_id
# #     if model.config.eos_token_id is None and tok.eos_token_id is not None:
# #         model.config.eos_token_id = tok.eos_token_id

# #     # optional checkpoint
# #     if args.ckpt and Path(args.ckpt).exists():
# #         state = torch.load(args.ckpt, map_location="cpu")
# #         model.load_state_dict(state, strict=False)

# #     model.to(device)
# #     model.eval()

# #     # 2) ReFT config
# #     component = args.component or f"gpt_neox.layers[{args.layer}].output"
# #     reft_config = ReftConfig(representations={
# #         "layer": args.layer,
# #         "component": component,
# #         "low_rank_dimension": args.rank,
# #         "intervention": pyreft.LoreftIntervention(
# #             embed_dim=model.config.hidden_size,
# #             low_rank_dimension=args.rank,
# #             dtype=next(model.parameters()).dtype,
# #             dropout=0.0,
# #             act_fn=None
# #         )
# #     })
# #     reft_model = pyreft.get_reft_model(model, reft_config)
# #     reft_model.set_device(device)
# #     return tok, reft_model, device

# # def print_module_tree(reft_model, limit=None):
# #     header("MODULE TREE (named_modules)")
# #     count = 0
# #     for name, mod in reft_model.named_modules():
# #         cls = type(mod).__name__
# #         tag = ""
# #         if isinstance(mod, pyreft.LoreftIntervention):
# #             tag = "  <-- LoReFT"
# #         print(f"{name if name else '<root>'} : {cls}{tag}")
# #         count += 1
# #         if limit and count >= limit:
# #             print(f"... (truncated at {limit} modules)")
# #             break

# # def print_parameters(reft_model, limit=None):
# #     header("PARAMETERS (named_parameters)")
# #     count = 0
# #     for name, p in reft_model.named_parameters():
# #         print(f"{name:<80} shape={tuple(p.shape)}  dtype={p.dtype}  device={p.device}  requires_grad={p.requires_grad}")
# #         count += 1
# #         if limit and count >= limit:
# #             print(f"... (truncated at {limit} parameters)")
# #             break

# # def inspect_loreft_details(reft_model):
# #     header("LoReFT DETAILS (rotate_layer.weight vs parametrizations.base)")
# #     lorefts = find_loreft_modules(reft_model)
# #     if not lorefts:
# #         print("No LoReFT modules found.")
# #         return

# #     for name, mod in lorefts:
# #         print(f"\n[LoReFT] {name}")
# #         # rotate_layer 基本信息
# #         print(f"  rotate_layer type: {type(mod.rotate_layer)}")

# #         # 正交后的实际权重
# #         try:
# #             R = mod.rotate_layer.weight
# #             print(f"  rotate_layer.weight                : shape={tuple(R.shape)}  dtype={R.dtype}  device={R.device}")
# #         except Exception as e:
# #             print("  (failed to read rotate_layer.weight)", e)

# #         # 底层 base 参数（正交参数化之前）
# #         try:
# #             orth = mod.rotate_layer.parametrizations.weight[0]
# #             base = orth.base
# #             print(f"  rotate_layer.parametrizations.weight[0] : {type(orth)}")
# #             print(f"  -> base                             : shape={tuple(base.shape)}  dtype={base.dtype}  device={base.device}")
# #         except Exception as e:
# #             print("  (failed to read parametrizations.weight[0].base)", e)

# #         # learned_source
# #         try:
# #             print(f"  learned_source.weight               : shape={tuple(mod.learned_source.weight.shape)}  dtype={mod.learned_source.weight.dtype}")
# #             if mod.learned_source.bias is not None:
# #                 print(f"  learned_source.bias                 : shape={tuple(mod.learned_source.bias.shape)}  dtype={mod.learned_source.bias.dtype}")
# #         except Exception as e:
# #             print("  (failed to read learned_source)", e)

# # def minimal_forward_and_keys(tok, reft_model, device):
# #     header("MINIMAL FORWARD (to inspect output keys)")
# #     # 构造一个最小 batch
# #     text = ["hello world"]
# #     batch = tok(text, padding=True, truncation=True, max_length=32, return_tensors="pt")
# #     batch["labels"] = torch.tensor([0], dtype=torch.long)
# #     batch = {k: v.to(device) for k, v in batch.items()}

# #     with torch.no_grad():
# #         out = reft_model(base=batch, return_dict=True)

# #     # 打印返回结构
# #     if hasattr(out, "keys"):
# #         try:
# #             keys = list(out.keys())
# #             print("reft_model(base=...).keys():", keys)
# #             for k in keys:
# #                 v = out[k]
# #                 print(f"  - {k}: {type(v)}")
# #                 if hasattr(v, "keys"):
# #                     print(f"    nested keys: {list(v.keys())}")
# #         except Exception as e:
# #             print("Failed to print dict-like keys:", e)
# #     else:
# #         print("Output type:", type(out))
# #         print(out)

# #     # 尝试通用抓取 loss/logits
# #     def get_outputs_from_reft(xx):
# #         if isinstance(xx, dict):
# #             for key in ("intervened_outputs", "counterfactual_outputs", "outputs", "model_outputs"):
# #                 if key in xx:
# #                     return xx[key]
# #             if "loss" in xx and "logits" in xx:
# #                 return xx
# #         return xx

# #     hf_out = get_outputs_from_reft(out)
# #     print("\n[Guess] extracted object:", type(hf_out))
# #     if isinstance(hf_out, dict):
# #         print("keys inside hf_out:", list(hf_out.keys()))
# #         if "loss" in hf_out:
# #             print("  -> loss found (dict):", hf_out["loss"])
# #         if "logits" in hf_out:
# #             print("  -> logits found (dict):", hf_out["logits"].shape)
# #     else:
# #         # 可能是 transformers 的输出对象
# #         if hasattr(hf_out, "loss"):
# #             print("  -> loss found (attr):", hf_out.loss)
# #         if hasattr(hf_out, "logits"):
# #             print("  -> logits found (attr):", hf_out.logits.shape)

# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--model", type=str, default="EleutherAI/pythia-410m", help="HF model name")
# #     ap.add_argument("--ckpt", type=str, default="", help="optional state_dict path to load (strict=False)")
# #     ap.add_argument("--layer", type=int, default=14, help="LoReFT layer index")
# #     ap.add_argument("--rank",  type=int, default=4,  help="LoReFT low-rank dimension")
# #     ap.add_argument("--component", type=str, default="", help="component string; default gpt_neox.layers[L].output")
# #     ap.add_argument("--limit-modules", type=int, default=0, help="print at most N modules (0=all)")
# #     ap.add_argument("--limit-params",  type=int, default=0, help="print at most N params (0=all)")
# #     ap.add_argument("--cpu", action="store_true", help="force cpu")
# #     args = ap.parse_args()

# #     tok, reft_model, device = build_model_and_reft(args)

# #     # 1) 打印设备、dtype
# #     header("ENV INFO")
# #     print("device =", device)
# #     print("model dtype =", next(reft_model.parameters()).dtype)

# #     # 2) 模块树
# #     print_module_tree(reft_model, limit=(args.limit_modules or None))

# #     # 3) 参数
# #     print_parameters(reft_model, limit=(args.limit_params or None))

# #     # 4) LoReFT 细节
# #     inspect_loreft_details(reft_model)

# #     # 5) 最小前向，打印返回键结构
# #     minimal_forward_and_keys(tok, reft_model, device)

# # if __name__ == "__main__":
# #     main()
# # from huggingface_hub import HfApi
# # api = HfApi()
# # files = api.list_repo_files("AlignmentResearch/robust_llm_clf_spam_pythia-1.4b_s-0_adv_tr_rt_t-0")
# # print(files)
# # from transformers import AutoModelForSequenceClassification, AutoTokenizer
# # model_id = "AlignmentResearch/robust_llm_clf_spam_pythia-1.4b_s-0_adv_tr_rt_t-0"
# # try:
# #     tok = AutoTokenizer.from_pretrained(model_id)
# #     model = AutoModelForSequenceClassification.from_pretrained(model_id)
# #     print("Loaded OK")
# # except Exception as e:
# #     print("Load failed:", e)

# import numpy as np
# uv = np.load("out/spam_pythia410m/probe_uv_ft_val.npz", allow_pickle=True)
# print(list(uv.keys()))
# v = uv["L14"]  # 或者你实际的 key
# u_np = uv.get("L14", None)
# print(u_np.shape)

# # embedding latent attack (FGSM or PGD)
# def latent_fgsm_single(
#     model,
#     tokenizer,
#     input_ids: torch.Tensor,
#     attention_mask: torch.Tensor,
#     true_label: int,
#     device: torch.device,
#     eps: float = 0.05
# ) -> Tuple[int, float, int, float, bool]:
#     """
#     Perform single-step FGSM attack in the embedding space.
#     """
#     model.eval()
#     ids = input_ids.unsqueeze(0).to(device)  # (1, L)
#     mask = attention_mask.unsqueeze(0).to(device)  # (1, L)
#     label_t = torch.tensor([true_label], device=device)  # (1,)
    
#     with torch.no_grad():
#         out = model(input_ids=ids, attention_mask=mask)
#         orig_logits = out.logits  # (1, num_labels)
#         orig_pred = int(orig_logits.argmax(-1).item())
#         orig_loss = float(F.cross_entropy(orig_logits, label_t).item())
        
#     # get embeddings and require grad
#     emb_layer = model.get_input_embeddings()
#     embeds = emb_layer(ids)  # [1,T,H]
#     embeds = embeds.clone().detach().requires_grad_(True)

#     out2 = model(inputs_embeds=embeds, attention_mask=mask)
#     loss = F.cross_entropy(out2.logits, label_t)
#     grad = torch.autograd.grad(loss, embeds)[0]  # [1,T,H]

#     # single-step sign update (L-inf eps)
#     adv_embeds = embeds + eps * torch.sign(grad)
#     with torch.no_grad():
#         out_adv = model(inputs_embeds=adv_embeds, attention_mask=mask)
#         adv_logits = out_adv.logits
#         adv_pred = int(adv_logits.argmax(-1).item())
#         adv_loss = float(F.cross_entropy(adv_logits, label_t).item())

#     success = (adv_pred != orig_pred)
#     return orig_pred, orig_loss, adv_pred, adv_loss, success


# def latent_pgd_single(
#     model,
#     tokenizer,
#     input_ids: torch.Tensor,
#     attention_mask: torch.Tensor,
#     true_label: int,
#     device: torch.device,
#     eps: float = 0.05,
#     steps: int = 20,
#     lr: float = None,
# ) -> Tuple[int, float, int, float, bool]:
#     """
#     PGD attack on inputs_embeds (L-inf projection).
#     """
#     model.eval()
#     ids = input_ids.unsqueeze(0).to(device)
#     mask = attention_mask.unsqueeze(0).to(device)
#     label_t = torch.tensor([true_label], device=device)

#     with torch.no_grad():
#         out = model(input_ids=ids, attention_mask=mask)
#         orig_logits = out.logits
#         orig_pred = int(orig_logits.argmax(-1).item())
#         orig_loss = float(F.cross_entropy(orig_logits, label_t).item())

#     emb_layer = model.get_input_embeddings()
#     emb0 = emb_layer(ids).detach()  # [1,T,H]
#     delta = torch.zeros_like(emb0, device=device).uniform_(-eps, eps)
#     delta = delta.requires_grad_(True)

#     if lr is None:
#         lr = eps / max(steps, 1) * 1.25

#     for _ in range(steps):
#         inputs_embeds = (emb0 + delta).requires_grad_(True)
#         outp = model(inputs_embeds=inputs_embeds, attention_mask=mask)
#         loss = F.cross_entropy(outp.logits, label_t)
#         grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
#         # ascent (untargeted)
#         delta = (delta + lr * torch.sign(grad)).detach()
#         # project to L-inf ball
#         delta = torch.clamp(delta, -eps, eps)
#         delta.requires_grad_(True)

#     with torch.no_grad():
#         final_embeds = emb0 + delta
#         out_adv = model(inputs_embeds=final_embeds, attention_mask=mask)
#         adv_logits = out_adv.logits
#         adv_pred = int(adv_logits.argmax(-1).item())
#         adv_loss = float(F.cross_entropy(adv_logits, label_t).item())

#     success = (adv_pred != orig_pred)
#     return orig_pred, orig_loss, adv_pred, adv_loss, success
# verify_hidden_vs_block_fixed.py
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# def verify(model_name="EleutherAI/pythia-410m", text="hello world, this is a small check."):
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     tok = AutoTokenizer.from_pretrained(model_name)
#     if tok.pad_token is None and tok.eos_token is not None:
#         tok.pad_token = tok.eos_token

#     model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
#     model.to(device).eval()

#     # ---- 1. Run forward once to get hidden_states ----
#     batch = tok(text, return_tensors="pt").to(device)
#     with torch.no_grad():
#         out = model(**batch, output_hidden_states=True, return_dict=True)
#     hidden_states = out.hidden_states  # tuple length = n_layers + 1
#     n_layers = len(hidden_states) - 1
#     print(f"Model has {n_layers} transformer blocks (len(hidden_states)={len(hidden_states)})")

#     # ---- 2. Try to locate the layer stack automatically ----
#     layer_stack = None
#     for cand in ["transformer.h", "transformer.layers", "model.layers", "gpt_neox.layers"]:
#         obj = model
#         try:
#             for attr in cand.split("."):
#                 obj = getattr(obj, attr)
#             # 检查是list/ModuleList
#             if hasattr(obj, "__getitem__"):
#                 layer_stack = obj
#                 print(f"Found layer stack at `{cand}` ({len(layer_stack)} layers)")
#                 break
#         except AttributeError:
#             continue

#     if layer_stack is None:
#         raise AttributeError("Cannot find layer stack automatically (tried common names).")

#     # ---- 3. Register forward hooks to capture block outputs ----
#     per_block_out = [None] * n_layers
#     hooks = []
#     for k in range(n_layers):
#         blk = layer_stack[k]
#         def make_hook(kk):
#             def hook_fn(module, inputs, output):
#                 per_block_out[kk] = output[0] if isinstance(output, (tuple, list)) else output
#             return hook_fn
#         hooks.append(blk.register_forward_hook(make_hook(k)))

#     # ---- 4. Forward again to trigger hooks ----
#     with torch.no_grad():
#         _ = model(**batch, output_hidden_states=False, return_dict=True)
#     for h in hooks: h.remove()

#     # ---- 5. Compare hidden_states[k+1] vs hook_out[k] ----
#     all_ok = True
#     for k in range(n_layers):
#         H_api = hidden_states[k+1]
#         H_hook = per_block_out[k]
#         if H_hook is None:
#             print(f"[WARN] layer {k} hook got None"); all_ok = False; continue
#         if H_api.shape != H_hook.shape:
#             print(f"[X] layer {k}: shape mismatch {H_api.shape} vs {H_hook.shape}")
#             all_ok = False; continue
#         max_abs = (H_api - H_hook).abs().max().item()
#         ok = max_abs < 1e-6
#         tag = "OK " if ok else "BAD"
#         print(f"[{tag}] layer k={k}: hidden_states[L={k+1}] vs block_out[k]; max|diff|={max_abs:.2e}")
#         all_ok = all_ok and ok

#     print("\nSummary:", "ALL GOOD ✅" if all_ok else "Some mismatches ❌")

# if __name__ == "__main__":
#     verify()
# import torch
# import pyreft
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from pyreft import ReftConfig

# # 加载模型和分词器
# model_name = "EleutherAI/pythia-160m"  # 使用一个较小的模型便于快速测试
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token  # 设置填充令牌

# # 修正ReFT配置
# reft_config = ReftConfig(
#     representations=[{  # representations 应为一个列表
#         "layer": 2,  # 干预第2层
#         "component": "block_output",  # 干预Transformer块的输出
#         "low_rank_dimension": 4,  # 低秩维度
#         "intervention": pyreft.LoreftIntervention(
#             embed_dim=model.config.hidden_size,  # 使用模型的隐藏层维度
#             low_rank_dimension=4,
#             dropout=0.0
#         )
#     }]
# )

# # 创建ReFT模型
# reft_model = pyreft.get_reft_model(model, reft_config)

# # 打印ReFT模型的可访问属性和方法进行检查
# print("ReFT模型的主要属性与方法:")
# for attr in dir(reft_model):
#     if not attr.startswith("_"):  # 过滤掉私有方法和属性
#         print(attr)


import torch
import pyreft
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyreft import ReftConfig

def test_activations_and_interventions():
    # 1. 加载模型和分词器
    model_name = "EleutherAI/pythia-160m"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. 配置多层ReFT干预
    reft_config = ReftConfig(
        representations=[
            {
                "layer": 1,
                "component": "block_output",
                "low_rank_dimension": 4,
                "intervention": pyreft.LoreftIntervention(
                    embed_dim=model.config.hidden_size,
                    low_rank_dimension=4,
                    dropout=0.0
                )
            },
            {
                "layer": 2, 
                "component": "block_output",
                "low_rank_dimension": 4,
                "intervention": pyreft.LoreftIntervention(
                    embed_dim=model.config.hidden_size,
                    low_rank_dimension=4,
                    dropout=0.0
                )
            }
        ]
    )

    # 3. 创建ReFT模型
    reft_model = pyreft.get_reft_model(model, reft_config)
    
    # 4. 准备测试输入
    text = "The capital of France is"
    inputs = tokenizer(text, return_tensors="pt").to(reft_model.model.device)
    
    print("=== 测试开始 ===")
    
    # 5. 测试 interventions 属性
    print(f"\n1. interventions 属性测试:")
    print(f"干预模块数量: {len(reft_model.interventions)}")
    
    for i, intervention in enumerate(reft_model.interventions):
        print(f"干预模块 {i}: {type(intervention)}")
        print(f"  - 可训练参数: {[name for name, param in intervention.named_parameters()]}")
        
        # 检查干预模块的具体参数
        for name, param in intervention.named_parameters():
            print(f"    {name}: {param.shape}")
    
    # 6. 前向传播并获取缓存激活
    print(f"\n2. get_cached_activations 测试:")
    
    # 启用激活缓存
    reft_model.return_collect_activations = True
    
    # 执行前向传播
    with torch.no_grad():
        outputs = reft_model(**inputs)
    
    # 尝试获取缓存的激活
    print("尝试获取缓存激活...")
    
    # 方法1: 使用 get_cached_activations (如果可用)
    if hasattr(reft_model, 'get_cached_activations'):
        try:
            # 获取特定层的激活
            activations_layer1 = reft_model.get_cached_activations(layer=1)
            activations_layer2 = reft_model.get_cached_activations(layer=2)
            
            print(f"第1层激活形状: {activations_layer1.shape}")
            print(f"第2层激活形状: {activations_layer2.shape}")
        except Exception as e:
            print(f"get_cached_activations 出错: {e}")
    
    # 方法2: 直接访问缓存属性
    print(f"\n3. 直接访问缓存属性:")
    
    # 检查各种可能的缓存属性
    cache_attrs = ['activations', 'hot_activations', 'get_cached_hot_activations']
    
    for attr in cache_attrs:
        if hasattr(reft_model, attr):
            value = getattr(reft_model, attr)
            print(f"{attr}: {type(value)}")
            
            # 如果是方法，尝试调用
            if callable(value):
                try:
                    result = value()
                    print(f"  {attr}() 返回: {type(result)}")
                    if hasattr(result, 'shape'):
                        print(f"  形状: {result.shape}")
                except Exception as e:
                    print(f"  {attr}() 调用出错: {e}")
            else:
                print(f"  值类型: {type(value)}")
    
    # 7. 测试手动干预应用
    print(f"\n4. 手动干预应用测试:")
    
    # 获取第一个干预模块
    if len(reft_model.interventions) > 0:
        intervention_module = reft_model.interventions[0]
        
        # 创建一些测试隐藏状态
        batch_size, seq_len, hidden_size = 2, 5, model.config.hidden_size
        test_hidden_state = torch.randn(batch_size, seq_len, hidden_size).to(reft_model.model.device)
        
        print(f"测试隐藏状态形状: {test_hidden_state.shape}")
        
        # 应用干预
        with torch.no_grad():
            intervened_hidden = intervention_module(test_hidden_state)
            print(f"干预后隐藏状态形状: {intervened_hidden.shape}")
            print(f"干预变化量: {torch.norm(intervened_hidden - test_hidden_state):.6f}")
    
    # 8. 测试在潜在对抗训练中的应用场景
    print(f"\n5. 潜在对抗训练场景演示:")
    
    def demonstrate_latent_adversarial_usage():
        """
        演示如何在潜在对抗训练中使用这些功能
        """
        print("潜在对抗训练步骤:")
        print("1. 获取特定层的隐藏表示 (使用 get_cached_activations)")
        print("2. 在该表示上添加对抗扰动")
        print("3. 通过干预模块处理扰动后的表示")
        print("4. 计算损失并反向传播")
        
        # 伪代码示例
        print("\n伪代码:")
        print("""
        # 前向传播获取隐藏表示
        reft_model.return_collect_activations = True
        outputs = reft_model(input_ids)
        h_l = reft_model.get_cached_activations(layer=target_layer)
        
        # 生成对抗扰动
        h_perturbed = h_l + epsilon * torch.randn_like(h_l)
        
        # 通过干预模块
        intervention = reft_model.interventions[layer_index]
        h_intervened = intervention(h_perturbed)
        
        # 计算损失 (需要自定义)
        loss = compute_adversarial_loss(h_intervened, labels)
        loss.backward()
        """)
    
    demonstrate_latent_adversarial_usage()
    
    print(f"\n=== 测试完成 ===")

if __name__ == "__main__":
    test_activations_and_interventions()
    
    
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import argparse
# from typing import List, Tuple, Optional

# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# # ----------------------------- utils: pretty print -----------------------------

# def print_section(title: str):
#     print("\n" + "=" * 80)
#     print(title)
#     print("=" * 80)

# def sample_named_modules(root: torch.nn.Module, max_print: int = 150) -> List[Tuple[str, torch.nn.Module]]:
#     items = list(root.named_modules())
#     print_section(f"[named_modules] total={len(items)} (show first {max_print})")
#     for i, (name, _) in enumerate(items[:max_print]):
#         print(f"{i:04d}: {name}")
#     return items

# def filtered_named_modules(root: torch.nn.Module, contains: List[str], max_print: int = 200) -> List[Tuple[str, torch.nn.Module]]:
#     items = []
#     for name, m in root.named_modules():
#         if all(s in name for s in contains):
#             items.append((name, m))
#     print_section(f"[filter] contains {contains} -> {len(items)} matches (show first {max_print})")
#     for i, (name, _) in enumerate(items[:max_print]):
#         print(f"{i:04d}: {name}")
#     return items

# # ----------------------- robust find / get helpers (demo) ----------------------

# def find_target_component_name(model_root: torch.nn.Module, model_type: str, layer_idx: int) -> str:
#     """
#     以“点号索引”返回一个**候选**的层块前缀，实际匹配用 get_module_by_name 兜底。
#     例如 GPT-NeoX: model.gpt_neox.layers.{i}
#     """
#     comp_idx = layer_idx - 1
#     if "gpt_neox" in model_type:
#         return f"model.gpt_neox.layers.{comp_idx}"
#     # 这里按需扩展其他结构：
#     # if "llama" in model_type: return f"model.layers.{comp_idx}"
#     # if "gpt2"   in model_type: return f"transformer.h.{comp_idx}"
#     # 默认：尝试最常见的 'layers.{i}'
#     return f"model.layers.{comp_idx}"

# def get_module_by_name(model_root: torch.nn.Module, module_name: str) -> torch.nn.Module:
#     """
#     先精确匹配 name==module_name；不行则用“包含匹配”，并选择最短名字（通常是层块本身）。
#     """
#     # exact
#     for name, module in model_root.named_modules():
#         if name == module_name:
#             return module
#     # contains
#     candidates = []
#     for name, module in model_root.named_modules():
#         if module_name in name:
#             candidates.append((name, module))
#     if candidates:
#         candidates.sort(key=lambda x: len(x[0]))
#         # 打印候选，便于你确认
#         print_section(f"[get_module_by_name] candidates for '{module_name}'")
#         for nm, _ in candidates[:20]:
#             print("  ->", nm)
#         return candidates[0][1]
#     raise ValueError(f"Module '{module_name}' not found under root '{type(model_root).__name__}'.")

# # ------------------------------ main debug routine -----------------------------

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model-name", type=str, required=True,
#                     help="HF model id or local path, e.g., EleutherAI/pythia-410m-deduped")
#     ap.add_argument("--layer-idx", type=int, default=14,
#                     help="1-based layer index you use for ReFT/attack")
#     ap.add_argument("--max-print", type=int, default=150)
#     ap.add_argument("--print-all", action="store_true", help="print full named_modules (may be huge)")
#     ap.add_argument("--run-forward", action="store_true",
#                     help="optionally run a tiny forward and hook to chosen module to print output shape")
#     ap.add_argument("--text", type=str, default="hello world")
#     args = ap.parse_args()

#     print_section("[load] model")
#     model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
#     model_type = getattr(model.config, "model_type", "")
#     print(f"model_type = {model_type}")

#     # 1) 打印所有模块（或前 N 个）
#     if args.print_all:
#         _ = sample_named_modules(model, max_print=10**9)
#     else:
#         _ = sample_named_modules(model, max_print=args.max_print)

#     # 2) 打印包含 'layers' 的模块
#     _ = filtered_named_modules(model, contains=["layers"], max_print=args.max_print)

#     # 3) 给出候选层块前缀，并打印该层下的子模块（如 .mlp/.attention/.output）
#     prefix = find_target_component_name(model, model_type, args.layer_idx)
#     print_section(f"[candidate prefix] for layer_idx={args.layer_idx}: '{prefix}'")

#     # 打印所有包含该前缀的模块名，帮助你看到真实结构
#     _ = filtered_named_modules(model, contains=[prefix], max_print=200)

#     # 4) 尝试常见的“层块本体”和“output子模块”两种获取方式
#     tried = []
#     for suffix in ["", ".output", ".mlp", ".attention", ".attn", ".self_attn"]:
#         target_name = prefix + suffix
#         try:
#             target_module = get_module_by_name(model, target_name)
#             print_section(f"[resolved] HIT -> '{target_name}'  (type={type(target_module).__name__})")
#             tried.append((target_name, target_module))
#         except Exception as e:
#             print(f"[resolved] MISS -> '{target_name}': {e}")

#     if not tried:
#         print_section("[result] No module resolved. Please inspect the printed names above and pick one manually.")
#         return

#     # 5) （可选）跑前向 + hook 一下，看看输出 tensor 的形状
#     if args.run_forward:
#         print_section("[forward + hook] try first resolved target")
#         target_name, target_module = tried[0]
#         print(f"hook target: {target_name} ({type(target_module).__name__})")

#         tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
#         if tok.pad_token is None:
#             tok.pad_token = tok.eos_token
#         model.config.pad_token_id = tok.pad_token_id

#         batch = tok([args.text], return_tensors="pt", padding=True, truncation=True, max_length=64)
#         batch["labels"] = torch.tensor([1])

#         container = {"shape": None, "dtype": None}

#         def hook_fn(module, inputs, output):
#             out = output[0] if isinstance(output, (tuple, list)) else output
#             if torch.is_tensor(out):
#                 container["shape"] = tuple(out.shape)
#                 container["dtype"] = out.dtype
#             else:
#                 container["shape"] = str(type(out))
#                 container["dtype"] = None

#         handle = target_module.register_forward_hook(hook_fn)
#         try:
#             with torch.no_grad():
#                 _ = model(**batch)
#         finally:
#             handle.remove()

#         print(f"[hook result] output shape: {container['shape']}, dtype: {container['dtype']}")
#         print("Done.")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare structure of reft_model (outer wrapper) vs reft_model.model (inner HF model)
"""

import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pyreft
from pyreft import ReftConfig


def summarize_named_modules(model, prefix, max_items=50):
    print(f"\n{'='*40}\n[ {prefix}.named_modules() ]")
    for i, (name, module) in enumerate(model.named_modules()):
        print(f"{i:03d}: {name:50s} | {type(module).__name__}")
        if i >= max_items:
            print(f"... (total {len(list(model.named_modules()))} modules, showing first {max_items})")
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default="EleutherAI/pythia-410m-deduped")
    ap.add_argument("--layer-idx", type=int, default=14)
    args = ap.parse_args()

    # 1️⃣ 载入原始 HF 模型
    print(f"\n[load HF model: {args.model_name}]")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model_type = getattr(model.config, "model_type", None)
    print(f"model type: {model_type}")

    summarize_named_modules(model, "HF model")

    # 2️⃣ 构造 ReFT 包装（最简配置）
    comp_idx = args.layer_idx - 1
    reft_config = ReftConfig(
        representations={
            "layer": args.layer_idx,
            "component": f"gpt_neox.layers[{comp_idx}].output",
            "low_rank_dimension": 4,
        }
    )

    reft_model = pyreft.get_reft_model(model, reft_config)

    # 3️⃣ 对比两层结构
    print("\n\n" + "="*80)
    print("✅ Comparing reft_model vs reft_model.model structure:")
    print("="*80)
    print(f"type(reft_model): {type(reft_model).__name__}")
    print(f"type(reft_model.model): {type(reft_model.model).__name__}")
    print(f"Keys in dir(reft_model): {[k for k in dir(reft_model) if not k.startswith('_')][:20]}")
    print(f"Keys in dir(reft_model.model): {[k for k in dir(reft_model.model) if not k.startswith('_')][:20]}")

    summarize_named_modules(reft_model, "reft_model")
    summarize_named_modules(reft_model.model, "reft_model.model")

    print("\n✅ 如果你想确认 attack hook 应该挂哪一层，请看 reft_model.model.named_modules() 输出。")
    print("   ReFT 自身模块（rotate_layer, learned_source等）在 reft_model.named_modules()。")
    print("   Hook 攻击层必须在 reft_model.model 中查找。")


if __name__ == "__main__":
    main()
