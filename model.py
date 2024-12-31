import os
from collections import OrderedDict

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    PeftModel,
)
from torch import nn as nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class HierachicalEncoder(nn.Module):
    def __init__(self, conf, raw_graph, features, sp_feature):
        super(HierachicalEncoder, self).__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device
        self.num_user = self.conf["num_users"]
        self.num_bundle = self.conf["num_bundles"]
        self.num_item = self.conf["num_items"]
        self.embedding_size = 64
        self.ui_graph, self.bi_graph_train, self.bi_graph_seen = raw_graph
        self.attention_components = conf["components"].split(", ") if conf[
                                                                          "components"] != "" else []

        self.content_feature, self.text_feature, self.cf_feature, self.bi_feature = features

        # MM >>>
        self.content_feature = nn.functional.normalize(
            self.content_feature, dim=-1)
        self.text_feature = nn.functional.normalize(self.text_feature, dim=-1)

        def dense(feature):
            module = nn.Sequential(OrderedDict([
                ('w1', nn.Linear(feature.shape[1], feature.shape[1])),
                ('act1', nn.ReLU()),
                ('w2', nn.Linear(feature.shape[1], 256)),
                ('act2', nn.ReLU()),
                ('w3', nn.Linear(256, 64)),
            ]))

            for m in module:
                init(m)
            return module

        # encoders for media feature
        self.c_encoder = dense(self.content_feature)

        self.multimodal_feature_dim = self.embedding_size
        # MM <<<

        # BI >>>
        self.bi_transformation = nn.Linear(
            self.embedding_size, self.embedding_size)
        init(self.bi_transformation)
        self.multimodal_feature_dim += self.embedding_size
        # BI <<<

        # UI >>>
        self.cf_transformation = nn.Linear(
            self.embedding_size, self.embedding_size)
        init(self.cf_transformation)

        self.multimodal_feature_dim += self.embedding_size
        # UI <<<

        # Multimodal Fusion:
        self.w_q = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_q)
        self.w_k = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_k)
        self.w_v = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_v)
        self.ln = nn.LayerNorm(self.embedding_size, elementwise_affine=False)

        if self.conf["sp_guided"]:
            self.w_q2 = nn.Linear(self.conf["llama_size"],
                                  self.embedding_size, bias=False)
            init(self.w_q2)
            self.w_k2 = nn.Linear(self.embedding_size,
                                  self.embedding_size, bias=False)
            init(self.w_k2)
            self.w_v2 = nn.Linear(self.embedding_size,
                                  self.embedding_size, bias=False)
            init(self.w_v2)
            self.ln2 = nn.LayerNorm(self.embedding_size, elementwise_affine=False)

    def selfAttention(self, features):
        # features: [bs, #modality, d]
        if "layernorm" in self.attention_components:
            features = self.ln(features)
        q = self.w_q(features)
        k = self.w_k(features)
        if "w_v" in self.attention_components:
            v = self.w_v(features)
        else:
            v = features
        # [bs, #modality, #modality]
        attn = q.mul(self.embedding_size ** -0.5) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)

        features = attn @ v  # [bs, #modality, d]
        # average pooling
        y = features.mean(dim=-2)  # [bs, d]

        return features, y

    def selfAttention2(self, sp_feature, features):
        # features: [bs, #modality, d]
        if "layernorm" in self.attention_components:
            features = self.ln2(features)
        q = self.w_q2(sp_feature.view(1, sp_feature.shape[0], -1))  # [1, 1, d]
        k = self.w_k2(features)  # [bs, #m, d]
        if "w_v" in self.attention_components:
            v = self.w_v2(features)
        else:
            v = features
        # [bs, 1, #modality]
        attn = q.mul(self.embedding_size ** -0.5) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)

        # [bs, 1, #modality]
        features = attn @ v  # [bs, 1, d]
        # average pooling

        y = features.mean(dim=-2)

        return y

    def forward(self, seq_modify, sp_feature, all=False):

        c_feature = self.c_encoder(self.content_feature)

        mm_feature_full = c_feature  # F.normalize(c_feature)
        mm_feature = mm_feature_full[seq_modify]  # [bs, n_token, d]

        features = [mm_feature]
        bi_feature_full = self.bi_transformation(self.bi_feature)
        bi_feature = bi_feature_full[seq_modify]
        features.append(bi_feature)

        cf_feature_full = self.cf_transformation(self.cf_feature)
        cf_feature = cf_feature_full[seq_modify]
        features.append(cf_feature)

        features = torch.stack(features, dim=-2)  # [bs, n_token, #modality, d]
        bs, n_token, N_modal, d = features.shape

        # multimodal fusion >>>
        feature_ori, final_feature = self.selfAttention(
            nn.functional.normalize(features.view(-1, N_modal, d), dim=-1))
        final_feature = final_feature.view(bs, n_token, d)
        # multimodal fusion <<<

        if self.conf["sp_guided"]:
            if self.conf["sp_sg"]:
                sp_feature = sp_feature.weight.detach()
            else:
                sp_feature = sp_feature.weight
            final_feature = self.selfAttention2(sp_feature, feature_ori)

        return final_feature


class BundleMLLM(nn.Module):
    def __init__(self, conf, raw_graph, features, item_info, is_training=False):
        super(BundleMLLM, self).__init__()
        self.conf = conf
        self.device = device = conf["device"]
        self.llama_model_path = conf["base_model"]

        self.ui_graph, self.bi_graph_train, self.bi_graph_seen = raw_graph
        self.content_feature, self.text_feature, self.cf_feature, self.bi_feature = features

        self.num_item = self.bi_graph_train.shape[1]

        self.item_info = item_info

        self.dataset_name = conf["dataset"]

        self.caption_key = "title_en"

        self.padding_side = "left"

        self.mode = conf["mode"]

        self.llama_model = AutoModelForCausalLM.from_pretrained(
            self.llama_model_path,
            load_in_8bit=True,
            torch_dtype=torch.bfloat16 if "Llama-2" in self.llama_model_path else torch.float16,
            device_map=self.device,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.llama_model_path, padding_side=self.padding_side)
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_prefix_space = False

        self.llama_model.resize_token_embeddings(len(self.tokenizer))

        # Projector
        self.mode = "text+mm"

        conf["llama_size"] = self.llama_model.config.hidden_size

        if self.conf["soft_prompt"]:
            self.num_pt = 1
            self.prompt_token = nn.Embedding(self.num_pt, self.llama_model.config.hidden_size).to(device)
            init(self.prompt_token)
        else:
            self.prompt_token = None
        self.fusion = HierachicalEncoder(conf, raw_graph, features, self.prompt_token)

        self.feat_dim = self.fusion.embedding_size

        self.projector = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(self.feat_dim, self.feat_dim * 2)),
            ('act1', nn.GELU()),
            ('output', nn.Linear(self.feat_dim * 2, self.llama_model.config.hidden_size)),
        ])).to(device)

        self.is_training = is_training
        if self.is_training:
            self.llama_model = prepare_model_for_int8_training(self.llama_model)

        if conf["from_pretrain"]:

            self.lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            if os.path.exists(f"{conf['pretrain_model_path']}/adapter_model.bin"):
                self.llama_model = PeftModel.from_pretrained(self.llama_model, conf["pretrain_model_path"],
                                                             config=self.lora_config, is_trainable=conf["train_lora"])
                print(">>>> pretrained llama lora loaded!")
            else:
                self.llama_model = get_peft_model(self.llama_model, self.lora_config)
                print(">>>> initial llama lora loaded!")

            if os.path.exists(f"{conf['pretrain_model_path']}/fusion.bin"):
                self.fusion.load_state_dict(torch.load(f"{conf['pretrain_model_path']}/fusion.bin"))
                print(">>>> fusion.bin loaded!")

            if self.conf["soft_prompt"] and os.path.exists(f"{conf['pretrain_model_path']}/fusion.bin"):
                self.prompt_token.load_state_dict(torch.load(f"{conf['pretrain_model_path']}/prompt_token.bin"))
                print(">>>> prompt_token.bin loaded!")

            if os.path.exists(f"{conf['pretrain_model_path']}/projector_model_fuse.bin"):
                self.projector.load_state_dict(torch.load(f"{conf['pretrain_model_path']}/projector_model_fuse.bin"))
                print(">>>> projector_model_fuse.bin loaded!")
        else:
            self.lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llama_model = get_peft_model(self.llama_model, self.lora_config)

        self.llama_model.model.config.use_cache = False

        self.llama_model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.llama_model.config.bos_token_id = 1
        self.llama_model.config.eos_token_id = 2

        #####################

        self.pad_embeds = self._embed_tokens(torch.tensor([self.tokenizer.pad_token_id]))
        self.bos_embeds = self._embed_tokens(torch.tensor([self.tokenizer.bos_token_id]))
        self.eos_embeds = self._embed_tokens(torch.tensor([self.tokenizer.eos_token_id]))

        #####################
        self._keys_to_ignore_on_save = []

        self.cutoff_len = 2048

    def _embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, 'model'):
            embeds = self.llama_model.base_model.base_model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    def generate_prompt(self, input_, target_):
        if self.dataset_name[:7] == "spotify":
            t_name = "playlist continuation"
            b_name = "music playlist"
            i_name = "song"
        else:
            t_name = "bundle construction"
            b_name = "fashion outfit"
            i_name = "fashion item"

        if "Llama-2-7b-chat-hf" in self.llama_model_path:
            chat = [
                {"role": "system",
                 "content": f"""You are a helpful and honest assistant. The following are multiple choice questions about {t_name}. You should directly answer the question by choosing the letter of the correct option. Only provide the letter of your answer, without any explanation or mentioning the option content."""},
                {"role": "user",
                 "content": f"""Given the partial {i_name}s: {input_}, which candidate {i_name} should be included into this {b_name}?\nOptions: {target_}\nYYour answer should indicate your choice with a single letter (e.g., “A,” “B,” “C,” etc.)."""},
                {"role": "assistant", "content": "The answer is "}
            ]
            return self.tokenizer.apply_chat_template(chat, tokenize=False).replace("<s>", "").replace("</s>", "")

        elif "Llama-2-7b-hf" in self.llama_model_path:
            return f"""You are a helpful and honest assistant. The following are multiple choice questions about {t_name}. You should directly answer the question by choosing the letter of the correct option. Only provide the letter of your answer, without any explanation or mentioning the option content. Question: Given the partial {b_name}: {input_}, which candidate {i_name} should be included into this {b_name}?\nOptions: {target_}\nYour answer should indicate your choice with a single letter (e.g., “A,” “B,” “C,” etc.).\nChoice: """

    def print_trainable_params(self):
        print('Trainable parameters:')
        for name, param in self.named_parameters():
            if param.requires_grad:
                print('\t' + name)

    def _gen_yes_or_no_ids(self, gt_label):
        label_ids = [3869 if float(gt_label) > 0.5 else 1939, 29889]
        return torch.Tensor(label_ids)

    def _gen_label_ids(self, gt_label, title):
        label_ids = self._tokenize(f"{chr(ord('A') + int(gt_label))}.", add_eos_token=False).input_ids

        return torch.Tensor(label_ids)

    def _tokenize(self, prompt, cutoff_len=None, add_eos_token=True):
        if cutoff_len is None:
            cutoff_len = self.cutoff_len
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )

        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
        return result

    def forward(self, gt_label, indices, candidates):
        inputs_embeds_list = []
        label_ids = []
        for i in range(len(indices)):
            attn = indices[i] != self.num_item
            input_indices = indices[i][:attn.sum()].tolist()
            input_candidates = candidates[i].tolist()
            input_ = ""
            target_ = ""
            for idx, j in enumerate(input_indices):
                str_dict = {
                    "mm": "content token: <Feature>" if not self.conf["soft_prompt"] else "<Feature>",  # soft_prompt
                    "text": self.item_info[str(int(j))][self.caption_key]
                }
                str_ = []
                for m in ["text", "mm", "ui", "bi"]:
                    if m in self.mode:
                        str_.append(str_dict[m])
                input_ += f"{idx + 1}. " + ", ".join(str_) + "; "

            for idx, j in enumerate(input_candidates):
                str_dict = {
                    "mm": "content token: <Feature>" if not self.conf["soft_prompt"] else "<Feature>",  # soft_prompt
                    "text": self.item_info[str(int(j))][self.caption_key]
                }
                str_ = []
                for m in ["text", "mm"]:
                    if m in self.mode:
                        str_.append(str_dict[m])
                target_ += f"{chr(ord('A') + idx)}. " + ", ".join(str_) + "; "

            prompt_emb = [self.bos_embeds.unsqueeze(0)]
            all_indices = torch.tensor(input_indices + input_candidates)
            all_proj_features = []
            if "mm" in self.mode:
                fused_features = self.fusion(all_indices.unsqueeze(0), self.prompt_token)
                all_proj_features.append(
                    self.projector(fused_features).squeeze(0)
                )
            count_feature_per = len(all_proj_features)

            if count_feature_per != 0:
                count_feature_all = count_feature_per * all_proj_features[0].shape[0]
                for idx, j in enumerate(self.generate_prompt(input_, target_).split("<Feature>")):
                    input_ids = torch.tensor(self._tokenize(j, add_eos_token=False).input_ids).to(self.device).long()
                    prompt_emb.append(self._embed_tokens(input_ids).unsqueeze(0))
                    if idx < count_feature_all:
                        if self.conf["soft_prompt"] and not self.conf["del_sp"]:
                            prompt_emb.append(self.prompt_token.weight.view(1, self.num_pt, -1))
                        prompt_emb.append(
                            all_proj_features[idx % count_feature_per][idx // count_feature_per].view(1, 1, -1))
            else:
                input_ids = torch.tensor(
                    self._tokenize(self.generate_prompt(input_, target_), add_eos_token=False).input_ids).to(
                    self.device).long()
                prompt_emb.append(self._embed_tokens(input_ids).unsqueeze(0))

            label = self._gen_label_ids(gt_label[i],
                                        self.item_info[str(int(input_candidates[gt_label[i]]))][self.caption_key]).to(
                self.device).long()
            prompt_emb.append(self._embed_tokens(label).unsqueeze(0))
            inputs_embeds_list.append(torch.cat(prompt_emb, dim=1))
            label_ids.append(label)

        emb_lens = [inputs_embeds.shape[1] for inputs_embeds in inputs_embeds_list]
        emb_max_len = max(emb_lens)
        wrapped_embs = self.pad_embeds.expand(len(inputs_embeds_list), emb_max_len, -1).clone()
        wrapped_atts = torch.zeros(len(inputs_embeds_list), emb_max_len, dtype=torch.long).to(self.device)
        for i, inputs_embeds in enumerate(inputs_embeds_list):
            if self.padding_side == "left":
                wrapped_embs[i, - emb_lens[i]:] = inputs_embeds
                wrapped_atts[i, - emb_lens[i]:] = 1
            else:
                wrapped_embs[i, :emb_lens[i]] = inputs_embeds
                wrapped_atts[i, :emb_lens[i]] = 1

        label_pad_ids = torch.full([len(wrapped_embs), emb_max_len], -100, dtype=torch.long).to(self.device)

        for i, label_id in enumerate(label_ids):
            if self.padding_side == "left":
                label_pad_ids[i, - label_id.shape[0]:] = label_id
            else:
                label_pad_ids[i, emb_lens[i] - label_id.shape[0]:emb_lens[i]] = label_id

        label_pad_ids = label_pad_ids.to(self.device)

        outputs = self.llama_model(
            inputs_embeds=wrapped_embs,
            attention_mask=wrapped_atts,
            return_dict=True,
            labels=label_pad_ids,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def generate(self, inputs_embeds_list, config):
        emb_lens = [inputs_embeds.shape[1] for inputs_embeds in inputs_embeds_list]
        emb_max_len = max(emb_lens)
        wrapped_embs = self.pad_embeds.expand(len(inputs_embeds_list), emb_max_len, -1).clone()
        wrapped_atts = torch.zeros(len(inputs_embeds_list), emb_max_len, dtype=torch.long).to(self.device)
        for i, inputs_embeds in enumerate(inputs_embeds_list):
            if self.padding_side == "left":
                wrapped_embs[i, (emb_max_len - emb_lens[i]):] = inputs_embeds
                wrapped_atts[i, (emb_max_len - emb_lens[i]):] = 1
            else:
                wrapped_embs[i, :emb_lens[i]] = inputs_embeds
                wrapped_atts[i, :emb_lens[i]] = 1

        self.token_lens = emb_lens

        return self.llama_model.generate(
            inputs_embeds=wrapped_embs,
            attention_mask=wrapped_atts,
            generation_config=config,
            return_dict_in_generate=True,
            output_scores=True,
        )

    def evaluate(self, indices, candidates, config):

        inputs_embeds_list = []
        for i in range(len(indices)):
            attn = indices[i] != self.num_item
            input_indices = indices[i][:attn.sum()].tolist()
            input_candidates = candidates[i].tolist()
            input_ = ""
            target_ = ""
            for idx, j in enumerate(input_indices):
                str_dict = {
                    "mm": "content token: <Feature>" if not self.conf["soft_prompt"] else "<Feature>",  # soft_prompt
                    "text": self.item_info[str(int(j))][self.caption_key]
                }
                str_ = []
                for m in ["text", "mm"]:
                    if m in self.mode:
                        str_.append(str_dict[m])
                input_ += f"{idx + 1}. " + ", ".join(str_) + "; "

            for idx, j in enumerate(input_candidates):
                str_dict = {
                    "mm": "content token: <Feature>" if not self.conf["soft_prompt"] else "<Feature>",  # soft_prompt
                    "text": self.item_info[str(int(j))][self.caption_key]
                }
                str_ = []
                for m in ["text", "mm"]:
                    if m in self.mode:
                        str_.append(str_dict[m])
                target_ += f"{chr(ord('A') + idx)}. " + ", ".join(str_) + "; "

            prompt_emb = [self.bos_embeds.unsqueeze(0)]
            all_indices = torch.tensor(input_indices + input_candidates)
            all_proj_features = []
            if "mm" in self.mode:
                fused_features = self.fusion(all_indices.unsqueeze(0), self.prompt_token)
                all_proj_features.append(
                    self.projector(fused_features).squeeze(0)
                )
            count_feature_per = len(all_proj_features)

            if count_feature_per != 0:
                count_feature_all = count_feature_per * all_proj_features[0].shape[0]
                for idx, j in enumerate(self.generate_prompt(input_, target_).split("<Feature>")):
                    input_ids = torch.tensor(self._tokenize(j, add_eos_token=False).input_ids).to(self.device).long()
                    prompt_emb.append(self._embed_tokens(input_ids).unsqueeze(0))
                    if idx < count_feature_all:
                        if self.conf["soft_prompt"] and not self.conf["del_sp"]:
                            prompt_emb.append(self.prompt_token.weight.view(1, self.num_pt, -1))
                        prompt_emb.append(
                            all_proj_features[idx % count_feature_per][idx // count_feature_per].view(1, 1, -1))
            else:
                input_ids = torch.tensor(
                    self._tokenize(self.generate_prompt(input_, target_), add_eos_token=False).input_ids).to(
                    self.device).long()
                prompt_emb.append(self._embed_tokens(input_ids).unsqueeze(0))

            inputs_embeds_list.append(torch.cat(prompt_emb, dim=1))

        return self.generate(inputs_embeds_list, config)


def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)
