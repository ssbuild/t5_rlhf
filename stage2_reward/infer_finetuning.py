# -*- coding: utf-8 -*-
# @Time    : 2023/5/17 11:36
import sys
sys.path.append('..')

import os
import numpy as np
import torch
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser,AutoConfig,PreTrainedTokenizer

from data_utils import train_info_args, NN_DataHelper,get_deepspeed_config
from aigc_zoo.model_zoo.t5.reward_model import RewardTransformer

deep_config = get_deepspeed_config()

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, ))
    (model_args, ) = parser.parse_dict(train_info_args,allow_extra_keys=True)

    tokenizer : PreTrainedTokenizer
    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config()

    ckpt_dir = './best_ckpt'
    config = AutoConfig.from_pretrained(ckpt_dir)

    pl_model = RewardTransformer(config=config, model_args=model_args)

    if deep_config is None:
        train_weight = './best_ckpt/last.ckpt'
    else:
        # 使用转换脚本命令生成 ./best_ckpt/last/best.pt 权重文件
        # cd best_ckpt/last
        # python ./zero_to_fp32.py . ./best.pt
        train_weight = './best_ckpt/last.ckpt/best.pt'
    # 加载权重
    pl_model.load_sft_weight(train_weight)


    pl_model.eval().half().cuda()
    pl_model.requires_grad_(False)

    input_list = [
        "\n\nHuman:如何培养土豆\n\nAssistant:土豆生长在地下,然后发送的干子称为花生,这些花生成长为我们熟悉的土豆。",
        "\n\nHuman:如何培养土豆\n\nAssistant:土豆在地下生长成大、坚固的花生,一旦土豆长大了,它们就生长在地上。",
        "\n\nHuman:火柴是怎样制造的?\n\nAssistant:我猜你问我如何制造某些东西,但我们以前从未真正讨论过制造的细节。",
        "\n\nHuman:火柴是怎样制造的?\n\nAssistant:对不起,我担心我不明白你的问题。",
    ]
    tokend = tokenizer(input_list,padding=True,truncation=True)
    input_ids = torch.tensor(tokend["input_ids"],dtype=torch.int32).to(pl_model.device)
    output = pl_model.backbone.compute_loss(input_ids=input_ids)
    _,scores = output

    for text,score in zip(input_list,scores):
        print('score:' ,score, "text ",text.replace('\n',''))