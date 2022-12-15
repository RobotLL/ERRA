
from functools import partial
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from data_util import _InputFeatures
import os
import torch
import copy
from torch import nn
from typing import *
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template
from openprompt.utils.logging import logger


# class _InputFeatures(InputFeatures):
    
#     tensorable_keys = ['img_embeds', 'input_ids', 'inputs_embeds', 'attention_mask', 'token_type_ids', 'label',
#         'decoder_input_ids', 'decoder_inputs_embeds', 'soft_token_ids',
#         'past_key_values', 'loss_ids']
#     all_keys = ['img_embeds','input_ids', 'inputs_embeds', 'attention_mask', 'token_type_ids', 'label',
#         'decoder_input_ids', 'decoder_inputs_embeds', 'soft_token_ids',
#         'past_key_values', 'loss_ids', 'guid', 'tgt_text', 'encoded_tgt_text', 'input_ids_len']
#     non_tensorable_keys = []    
    
    
#     def __init__(self,
#                 img_embeds: Optional=None,
#                 input_ids: Optional[Union[List, torch.Tensor]] = None,
#                 inputs_embeds: Optional[torch.Tensor] = None,
#                 attention_mask: Optional[Union[List[int], torch.Tensor]] = None,
#                 token_type_ids: Optional[Union[List[int], torch.Tensor]] = None,
#                 label: Optional[Union[int, torch.Tensor]] = None,
#                 decoder_input_ids: Optional[Union[List, torch.Tensor]] = None,
#                 decoder_inputs_embeds: Optional[torch.Tensor] = None,
#                 soft_token_ids: Optional[Union[List, torch.Tensor]] = None,
#                 past_key_values: Optional[torch.Tensor] = None,  # for prefix_tuning
#                 loss_ids: Optional[Union[List, torch.Tensor]] = None,
#                 guid: Optional[str] = None,
#                 tgt_text: Optional[str] = None,
#                 use_cache: Optional[bool] = None,
#                 encoded_tgt_text: Optional[str] = None,
#                 input_ids_len: Optional[int] = None,
#                 **kwargs
#                 ):
#         InputFeatures.__init__(self,input_ids, inputs_embeds, attention_mask, token_type_ids, label, decoder_input_ids, decoder_inputs_embeds, soft_token_ids, past_key_values,loss_ids, guid, tgt_text, use_cache, encoded_tgt_text, input_ids_len, **kwargs)
#         self.img_embeds = img_embeds



class PrefixTuningTemplate(Template):
    r"""This is the implementation which support T5 and other Encoder-Decoder model,
    as soon as their blocks allows the ``past_key_values`` to be injected to the model.
    This implementation modifies the huggingface's T5 forward without touching the code-base.
    However, it may fail to work when used in DataParallel model. Please use it using
    single gpu or model-parallel training.
    Args:
        model (:obj:`PreTrainedModel`): The pre-trained model.
        plm_config (:obj:`PretrainedConfig`): The configuration of the current pre-trained model.
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model.
        mapping_hook (:obj:`nn.Module`, optional):
        text (:obj:`str`, optional): 
        mask_token (:obj:`str`, optional):
        num_token (:obj:`int`, optional):
        num_token_per_rel (:obj:`int`,o ptional): The number of soft prompt tokens for each relation/attribution
        placeholder_mapping (:obj:`dict`):
        prefix_dropout (:obj:`float`, optional): The dropout rate for the prefix sequence.
    """
    registered_inputflag_names = ["loss_ids", 'shortenable_ids']

    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 mapping_hook: Optional[nn.Module] = None,
                 text: Optional[str] = None,
                 mask_token: str = '<mask>',
                 num_token: Optional[int] = 5,    #number of prefix tokens
                 placeholder_mapping: dict = {'<text_a>':'text_a', '<text_b>':'text_b'},
                 prefix_dropout: Optional[float] = 0.0,
                 mid_dim: Optional[int] =  512,
                 num_attr: Optional[int] = 0,
                 num_token_per_rel: Optional[int] = 1,
                 random_range: Optional[float] = 0.5,
                 using_encoder_past_key_values: Optional[bool] = True,
                 using_decoder_past_key_values: Optional[bool] = True,
                 **param
                ):
        super().__init__(tokenizer=tokenizer,
                         #mask_token=mask_token,
                         placeholder_mapping=placeholder_mapping)
        raw_embedding = model.get_input_embeddings()
        self.raw_embedding = raw_embedding
        self.config = model.config
        self.mapping_hook = mapping_hook
        self.random_range = random_range        
        self.embedding_size = raw_embedding.weight.shape[-1]
        self.num_token = num_token
        self.num_attr = num_attr
        self.num_token_per_rel=num_token_per_rel
        self.using_encoder_past_key_values = using_encoder_past_key_values
        self.using_decoder_past_key_values = using_decoder_past_key_values
        assert (self.using_encoder_past_key_values or self.using_decoder_past_key_values), "Can't be both False."
        if not self.config.is_encoder_decoder and not self.using_decoder_past_key_values:
            logger.warning("Ignore using_decoder_past_key_values=False in a decoder-only LM.")

        if isinstance(self.config, T5Config):
            self.n_layer = self.config.num_layers
            self.n_embd = self.config.d_model
            self.n_head = self.config.num_heads
            self.n_decoder_layer = self.config.num_decoder_layers
            self.match_n_decoder_layer = self.n_decoder_layer
            self.match_n_layer = self.n_layer
        elif isinstance(self.config, GPT2Config):
            self.n_decoder_layer = self.config.n_layer
            self.n_embd = self.config.n_embd
            self.n_head = self.config.n_head
            self.match_n_decoder_layer = self.n_decoder_layer
        self.mid_dim = mid_dim
        self.match_n_head = self.n_head
        self.match_n_embd = self.n_embd // self.n_head
        self.prefix_dropout = prefix_dropout
        self.dropout = nn.Dropout(self.prefix_dropout)

        self.default_text1 = '{"placeholder": "text_a"} {"mask"}'
        self.default_text2 = '{"placeholder": "text_a"} {"placeholder": "text_b"} {"mask"}'

        self.text = text
        
        self.generate_parameters() # in prefix tuning the template text has no interact with the parameters.

        self.plm_modified = False # flag to indicate whether the function of plm are replaced for prefix tuning.
        self.modify_plm(model)
    

    def on_text_set(self):
        self.text = self.parse_text(self.text)
        self.generate_parameters()


    def get_past_key_values(self, batch_size=1):
        pvs = []
        if self.config.is_encoder_decoder and self.using_encoder_past_key_values:
            input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1)
            temp_control = self.wte(input_tokens)
            past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
            _, seqlen, _ = past_key_values.shape
            past_key_values = past_key_values.view(batch_size, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                self.match_n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            pvs.append(past_key_values)
        else:
            pvs.append(None)

        if (not self.config.is_encoder_decoder) or self.using_decoder_past_key_values:
            decoder_input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1)
            decoder_temp_control = self.decoder_wte(decoder_input_tokens)
            decoder_past_key_values = self.decoder_control_trans(decoder_temp_control) #bsz, seqlen, layer*emb
            _, decoder_seqlen, _ = decoder_past_key_values.shape
            decoder_past_key_values = decoder_past_key_values.view(batch_size, decoder_seqlen, self.match_n_decoder_layer * 2, self.match_n_head,
                                                self.match_n_embd)
            decoder_past_key_values = self.dropout(decoder_past_key_values)
            decoder_past_key_values = decoder_past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            pvs.append(decoder_past_key_values)
        else:
            pvs.append(None)
        return pvs

    def generate_parameters(self) -> None:
        r"""
        Generate parameters needed for new tokens' embedding in P-tuning
        """
        
        self.input_tokens = nn.Parameter(torch.arange(self.num_token).long(), requires_grad=False) # to allow automatic devicing
        if self.config.is_encoder_decoder and self.using_encoder_past_key_values:
            self.wte = nn.Embedding(self.num_token, self.n_embd)
            self.control_trans = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                # nn.Linear(self.mid_dim, self.mid_dim),
                # nn.Tanh(),
                nn.Linear(self.mid_dim, self.n_layer * 2 * self.n_embd))

        if (not self.config.is_encoder_decoder) or self.using_decoder_past_key_values:
            self.decoder_wte = nn.Embedding(self.num_token, self.n_embd)
            self.decoder_control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            # nn.Linear(self.mid_dim, self.mid_dim),
            # nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_decoder_layer * 2 * self.n_embd))
        
        # Generate the embedding needed for soft tokens

        self.soft_embedding = nn.Embedding(4, self.embedding_size)  #for tactile signal





#     def wrap_one_example(self, example) -> List[Dict]:
#         if self.text is None:
#             if example.text_b is None:
#                 self.text = self.default_text1
#             else:
#                 self.text = self.default_text2
#         return super().wrap_one_example(example)

  
    def wrap_one_example(self, example) -> List[Dict]:
        text = self.text
        wrapped_parts_to_tokenize = []
        tmp_dict = {}
        relations = None
        for d in text:
            if 'placeholder' in d:
                tmp_dict['text']=d["add_prefix_space"] + d.get("post_processing", lambda x:x)(getattr(example, d['placeholder']))
                tmp_dict['loss_ids'] = 0
                tmp_dict['shortenable_ids'] = 1 
                tmp_dict['soft_token_ids'] = 0
                wrapped_parts_to_tokenize.append(tmp_dict)
                tmp_dict = {}
            elif 'image' in d:
                tmp_dict['text']=''
                tmp_dict['loss_ids'] = 0
                tmp_dict['shortenable_ids'] = 0 
                tmp_dict['soft_token_ids'] = 3
                wrapped_parts_to_tokenize.append(tmp_dict)
                tmp_dict = {}     
            elif 'tactile' in d:
                tmp_dict['text']=''
                tmp_dict['loss_ids'] = 0
                tmp_dict['shortenable_ids'] = 0 
                tmp_dict['soft_token_ids'] =example.meta['tact']+1   ######这里要根据processor改 需要取值1,2
                wrapped_parts_to_tokenize.append(tmp_dict)
                tmp_dict = {}                
                
            elif 'mask' in d:
                tmp_dict['text']='<mask>'
                tmp_dict['loss_ids'] = 1
                tmp_dict['shortenable_ids'] = 0 
                tmp_dict['soft_token_ids'] = 0
                wrapped_parts_to_tokenize.append(tmp_dict)
                tmp_dict = {}
            elif 'special' in d:
                tmp_dict['text']=d['special']
                tmp_dict['loss_ids'] = 0
                tmp_dict['shortenable_ids'] = 0
                tmp_dict['soft_token_ids'] = 0
                wrapped_parts_to_tokenize.append(tmp_dict)
                tmp_dict = {}                
            elif 'text' in d:
                tmp_dict['text']=d["add_prefix_space"] + d['text']
                tmp_dict['loss_ids'] = 0
                tmp_dict['shortenable_ids'] = 1  
                tmp_dict['soft_token_ids'] = 0
                wrapped_parts_to_tokenize.append(tmp_dict)
                tmp_dict = {}                 
            else:
                raise ValueError(f'can not parse {d}') 
                
            wrapped_parts_not_tokenize = {'guid':example.guid, 'tgt_text':example.tgt_text, 'img_embeds': example.meta['img']} #{'guid':example.guid, 'tgt_text':example.tgt_text, 'relations':relations}
        return [wrapped_parts_to_tokenize ,wrapped_parts_not_tokenize ]
    def expand_to_batchsize(self, tup,  batch_size):
        return tuple(t.expand(-1, batch_size,-1,-1,-1) for t in tup)

    def expand_to_batchsize_for_layer(self, tup,  batch_size, layer_id):
        return tup[layer_id].expand(-1, batch_size,-1,-1,-1)



    def process_batch(self, batch: Union[Dict, _InputFeatures]) -> Union[Dict, _InputFeatures]:
        r"""
        Convert input_ids to inputs_embeds
        for normal token, use the embedding inside PLM
        for new token, use MLP or LSTM
        """
        
        
        batch_size = batch['input_ids'].size(0)
        self.past_key_values = self.get_past_key_values()

        if self.config.is_encoder_decoder: 
            # the attention_mask is encoder attention mask, the new token mask will be added in modified_encoder_forward.
            pass
        else: # the attention_mask is decoder attention mask
            past_key_values = self.expand_to_batchsize(self.past_key_values[1], batch_size)
            if 'attention_mask' in batch:
                am = batch['attention_mask']
                batch['attention_mask'] = torch.cat([torch.ones((batch_size,self.num_token), dtype = am.dtype,device=am.device), am], dim=-1)
            batch['past_key_values'] = past_key_values
            
        # replace the soft tokens

        raw_embeds = self.raw_embedding(batch['input_ids'])
        
        # broadcast img_embeds

        img_embeds = batch['img_embeds']
        img_embeds = torch.cat([img_embeds, torch.zeros(raw_embeds.size(0),raw_embeds.size(-1)-img_embeds.size(-1),device=raw_embeds.device)],-1)
        img_embeds = img_embeds.repeat(1,raw_embeds.size(1)).reshape(raw_embeds.shape)
        img_embeds = img_embeds.type(torch.float32)
   
        soft_token_ids = batch['soft_token_ids']
        soft_embeds = self.soft_embedding(batch['soft_token_ids'])
        inputs_embeds = torch.where((soft_token_ids > 0).unsqueeze(-1), soft_embeds, raw_embeds)
        inputs_embeds = torch.where((soft_token_ids == 3).unsqueeze(-1), img_embeds, inputs_embeds)
        ###这里需要debug
        

        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        return batch
    
    def modify_plm(self, model):
        if self.plm_modified:
            return None
        if isinstance(model, T5ForConditionalGeneration):
            if self.using_encoder_past_key_values:
                backup_encoder_forward_functions = []
                for i, layer_module in enumerate(model.encoder.block):
                    backup_encoder_forward_functions.append(layer_module.layer[0].forward)
                    def modified_encoder_forward(*args, **kwargs):
                        layer_id = kwargs.pop('layer_id')
                        batch_size = args[0].shape[0]
                        device = args[0].device
                        if kwargs['past_key_value'] is None:
                            kwargs['past_key_value'] =self.expand_to_batchsize_for_layer(self.past_key_values[0], batch_size, layer_id).to(device)
                        if kwargs['attention_mask'] is not None:
                            am = kwargs['attention_mask']  # Should check the format of the attention_mask when moving to a new plm.
                            kwargs['attention_mask'] = torch.cat([-torch.zeros((*am.shape[:-1],self.num_token), dtype = am.dtype,device=am.device), am], dim=-1)
                        return backup_encoder_forward_functions[layer_id](*args, **kwargs)
                    layer_module.layer[0].forward = partial(modified_encoder_forward, layer_id=i)

            if self.using_decoder_past_key_values:
                backup_decoder_self_attn_forward_functions = []
                for i, layer_module in enumerate(model.decoder.block):
                    backup_decoder_self_attn_forward_functions.append(layer_module.layer[0].forward)
                    def modified_decoder_self_attn_forward(*args, **kwargs):
                        batch_size = args[0].shape[0]
                        layer_id = kwargs.pop('layer_id')
                        device = args[0].device
                        if kwargs['past_key_value'] is None:
                            kwargs['past_key_value'] = self.expand_to_batchsize_for_layer(self.past_key_values[1], batch_size, layer_id).to(device)
                        if kwargs['past_key_value'][0].size(-2) + args[0].size(-2) == kwargs['attention_mask'].size(-1):
                            pass
                        elif kwargs['past_key_value'][0].size(-2) + args[0].size(-2) == kwargs['attention_mask'].size(-1) +self.num_token:
                            am = kwargs['attention_mask']
                            kwargs['attention_mask'] = torch.cat([torch.zeros((*am.shape[:-1],self.num_token), dtype = am.dtype,device=am.device), am], dim=-1)
                        else:
                            raise RuntimeError("Size not match: past length: {}, inputlength:{},\
                                attention mask length {}".format(kwargs['past_key_value'][0].size(-2), 
                                args[0].size(-2),kwargs['attention_mask'].size(-1)))
                        
                        return backup_decoder_self_attn_forward_functions[layer_id](*args, **kwargs)
                    layer_module.layer[0].forward = partial(modified_decoder_self_attn_forward, layer_id=i)

        elif isinstance(model, GPT2LMHeadModel):
            pass
        else:
            raise NotImplementedError
        self.plm_modified = True


