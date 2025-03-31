import torch
import torch.nn.functional as F
import json
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def get_expert_masks():
    """
    创建钩子函数来记录MoE模型中专家的选择情况

    Returns:
        tuple: (专家选择记录列表, 钩子函数)
    """
    expert_masks = []

    def expert_mask_hook(module, input, output):
        """记录专家选择的钩子函数"""
        # 检查输出格式是否符合预期
        if isinstance(output, tuple) and len(output) > 1:
            hidden_states, router_logits = output

            # 获取批次大小和序列长度
            # [batch_size, seq_len, hidden_dim]
            if len(hidden_states.shape) == 3:
                batch_size, sequence_length, hidden_dim = hidden_states.shape
            else:  # [batch_size * seq_len, hidden_dim]
                batch_size = 1
                sequence_length = hidden_states.shape[0]
                hidden_dim = hidden_states.shape[1]
                hidden_states = hidden_states.unsqueeze(0)  # 添加批次维度

            # 获取模块的专家数量和top-k
            num_experts = module.num_experts if hasattr(
                module, "num_experts") else 8
            top_k = module.top_k if hasattr(module, "top_k") else 2

            # 计算路由权重和专家选择
            routing_weights = F.softmax(
                router_logits, dim=1, dtype=torch.float)
            topk_weights, selected_experts = torch.topk(
                routing_weights, top_k, dim=-1)

            # 创建专家掩码
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=num_experts).permute(2, 1, 0)

            # 收集信息
            layer_name = next((name for name, mod in module.named_modules()
                              if mod is module), "unknown_layer")
            expert_masks.append({
                'layer_idx': len(expert_masks),
                'layer_name': layer_name,
                'selected_experts': selected_experts.reshape(batch_size, sequence_length, top_k).detach().cpu().numpy(),
                'routing_weights': topk_weights.reshape(batch_size, sequence_length, top_k).detach().cpu().numpy(),
                'expert_mask': expert_mask.detach().cpu().numpy()
            })
        return output

    return expert_masks, expert_mask_hook


def export_expert_masks_to_file(expert_masks, tokenizer, output_ids, output_file):
    """
    将专家选择信息导出到文件

    Args:
        expert_masks: 专家选择记录列表
        tokenizer: 分词器，用于解码token
        output_ids: 模型输出的token ID序列
        output_file: 输出文件路径
    """
    export_data = {
        "summary": {
            "total_layers": len(expert_masks),
            "experts_per_layer": {},
            "global_expert_usage": defaultdict(int)
        },
        "token_level_analysis": [],
        "layer_level_analysis": []
    }

    # 解析输出tokens
    try:
        if hasattr(tokenizer, "batch_decode"):
            tokens = tokenizer.batch_decode(
                output_ids, skip_special_tokens=False)
        else:
            # 兼容不同tokenizer的API
            tokens = [tokenizer.decode(token_id) for token_id in output_ids]
    except:
        tokens = ["<token_decode_error>" for _ in range(len(output_ids))]

    # 解析每层的专家选择
    for layer_idx, layer_data in enumerate(expert_masks):
        layer_experts = layer_data['selected_experts']
        layer_weights = layer_data['routing_weights']

        # 统计该层专家使用情况
        expert_usage = defaultdict(int)
        for batch in range(layer_experts.shape[0]):
            for seq in range(layer_experts.shape[1]):
                for k in range(layer_experts.shape[2]):
                    expert_idx = int(layer_experts[batch, seq, k])
                    expert_usage[expert_idx] += 1
                    export_data["summary"]["global_expert_usage"][str(
                        expert_idx)] += 1

        export_data["summary"]["experts_per_layer"][f"layer_{layer_idx}"] = dict(
            expert_usage)

        # 记录该层的详细信息
        export_data["layer_level_analysis"].append({
            "layer_idx": layer_idx,
            "layer_name": layer_data.get('layer_name', f"layer_{layer_idx}"),
            "expert_usage": dict(expert_usage),
            "total_tokens_processed": layer_experts.shape[0] * layer_experts.shape[1]
        })

    # 记录token级别的专家选择
    seq_len = min(len(output_ids),
                  expert_masks[0]['selected_experts'].shape[1])
    for pos in range(seq_len):
        token_data = {
            "position": pos,
            "token": tokens[pos] if pos < len(tokens) else "<unknown>",
            "token_id": int(output_ids[pos]) if pos < len(output_ids) else -1,
            "experts_by_layer": {}
        }

        for layer_idx, layer_data in enumerate(expert_masks):
            if pos < layer_data['selected_experts'].shape[1]:
                experts = layer_data['selected_experts'][0, pos].tolist()
                weights = layer_data['routing_weights'][0, pos].tolist()
                token_data["experts_by_layer"][f"layer_{layer_idx}"] = {
                    "selected_experts": experts,
                    "routing_weights": weights
                }

        export_data["token_level_analysis"].append(token_data)

    # 保存到文件
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"专家选择信息已保存至: {output_file}")

    return export_data
