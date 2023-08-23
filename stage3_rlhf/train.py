# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 17:08
import sys
sys.path.append('..')

import copy
import logging
import math
import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.trainer.pl.modelcheckpoint import FabricModelCheckpoint
from lightning.fabric.strategies import DeepSpeedStrategy
from transformers import HfArgumentParser
from data_utils import NN_DataHelper, train_info_args, get_deepspeed_config,global_args

from aigc_zoo.model_zoo.t5.ppo_model import MyPPOTransformer,LoraArguments,LoraConfig,PPOArguments,PPOConfig
from reward_weight import load_reward_model, load_ref_model
from deep_training.nlp.rl.ppo.ppo_trainer import PPOTrainer


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments, PPOArguments))
    model_args, training_args, data_args, lora_args, ppo_args = parser.parse_dict(train_info_args)
    lora_args = lora_args.config
    ppo_args = ppo_args.config

    output_weight_dir = './best_ckpt'

    dataHelper = NN_DataHelper(model_args, training_args, data_args, ppo_args=ppo_args)
    config_kwargs = {"torch_dtype": torch.float16}
    if global_args["num_layers"] > 0:
        if global_args["num_layers"] > 0:
            config_kwargs["num_layers"] = global_args["num_layers"]
            config_kwargs["num_decoder_layers"] = global_args["num_layers"]
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs=config_kwargs)

    dataHelper.make_dataset_all()

    deepspeed_config = get_deepspeed_config()
    strategy = 'ddp' if torch.cuda.device_count() >= 1 else 'auto'
    if deepspeed_config is not None and len(deepspeed_config):
        strategy = DeepSpeedStrategy(config=deepspeed_config, )

    checkpoint_callback = FabricModelCheckpoint(
        # monitor="loss",
        dirpath=output_weight_dir,
        save_weights_only=True,
        every_n_epochs=1,
        every_n_train_steps=1000 // training_args.gradient_accumulation_steps,
        save_last=True,
        # 模型参数
        model_args=model_args,
        training_args=training_args,
        lora_args=lora_args, )


        
    trainer = PPOTrainer(
        callbacks=[ checkpoint_callback],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,
        checkpoint_dir=data_args.output_dir,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        strategy=strategy,
        precision='16',# 可以自行尝试  "32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"
    )


    if trainer.global_rank == 0:
        #加载 lora reward 权重
        pl_reward_model = load_reward_model('../stage2_reward/best_ckpt/last')

        # 加载 finetuning 权重
        # pl_reward_model = load_reward_model('../stage2_reward/best_ckpt','../stage2_reward/best_ckpt/last/best.pt')

        reward_device = torch.cuda.device_count() - 1
        pl_reward_model = pl_reward_model.to(reward_device)
        delta_reward = True

        def get_reward(input_data,output_data):
            inputs = tokenizer(
                input_data,
                padding=True,
                truncation=True,
                max_length=data_args.max_seq_length,
                return_attention_mask=False,
                return_tensors="pt",
            ).to(reward_device)

            outputs = tokenizer(
                output_data,
                padding=True,
                truncation=True,
                max_length=data_args.max_seq_length,
                return_attention_mask=False,
                return_tensors="pt",
            ).to(reward_device)


            out = []
            for i in range(len(inputs)):
                batch_ixs = slice(i * 1, (i + 1) * 1)
                input_ids = inputs.input_ids[batch_ixs]
                decoder_input_ids = outputs.input_ids[batch_ixs]
                rewards = pl_reward_model.forward_returns(**{
                    "input_ids": input_ids,"decoder_input_ids": decoder_input_ids,
                })
                out.extend(rewards)
            return torch.hstack(out)

        def reward_fn(prompts,outputs, org_labels, **kwargs):
            org_labels = [str(l, encoding='utf-8') for l in org_labels]
            rewards = get_reward(prompts,outputs)
            if not delta_reward:
                return rewards

            original_rewards = get_reward(prompts, org_labels)
            return rewards - original_rewards
    else:
        reward_fn = None


    pl_model = MyPPOTransformer(config=config,model_args=model_args,training_args=training_args,lora_args=lora_args,ppo_args=ppo_args,
                                quantization_config=global_args["quantization_config"],
                                load_in_8bit=global_args["load_in_8bit"],
                                device_map={"": trainer.local_rank} if trainer.world_size > 1 else "auto",
                                torch_dtype=torch.float16,
                                new_num_tokens=len(tokenizer),  # 可能扩充词
                                )

    config.save_pretrained(output_weight_dir)

    #加载权重继续训练
    #pl_model.load_sft_weight('best_ckpt/best.pt',is_trainable=True)

    pl_model.float()


    # pl_ref_model = load_ref_model('../stage2_reward/best_ckpt')
    pl_ref_model = copy.deepcopy(pl_model)
    pl_ref_model.eval().half()
    pl_ref_model.requires_grad_(False)

    train_datasets = dataHelper.load_distributed_random_sampler(
        dataHelper.train_files,
        with_load_memory=True,
        collate_fn=dataHelper.collate_fn,
        # batch_size=training_args.train_batch_size,
        batch_size=ppo_args.chunk_size,
        drop_last=True,  # 多卡建议扔掉
        num_processes=trainer.world_size, process_index=trainer.global_rank)

    if train_datasets is not None:
        trainer.fit(pl_model,
                    ref_model=pl_ref_model,
                    train_loader=train_datasets,
                    tokenizer=tokenizer,
                    reward_fn=reward_fn,
                    ppo_config=ppo_args,
                    stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
                    )
