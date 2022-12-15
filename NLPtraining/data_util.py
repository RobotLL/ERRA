import os
import csv
import torch
import pandas as pd
from tqdm.std import tqdm
from random import shuffle
from torch.utils.data import Dataset
from typing import List, Dict, Callable
from collections import OrderedDict
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from openprompt.utils import round_list, signature
from openprompt.prompt_base import Template, Verbalizer
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt.plms.utils import TokenizerWrapper
from openprompt.data_utils import InputFeatures, InputExample
import numpy as np
from typing import *
from abc import abstractmethod


class DataProcessor:
    """
    labels of the dataset is optional 
    here's the examples of loading the labels:
    :obj:`I`: ``DataProcessor(labels = ['positive', 'negative'])``
    :obj:`II`: ``DataProcessor(labels_path = 'datasets/labels.txt')``
    labels file should have label names seperated by any blank characters, such as
    ..  code-block:: 
        positive neutral
        negative
    Args:
        labels (:obj:`Sequence[Any]`, optional): class labels of the dataset. Defaults to None.
        labels_path (:obj:`str`, optional): Defaults to None. If set and :obj:`labels` is None, load labels from :obj:`labels_path`. 
    """

    def __init__(self,
                 labels: Optional[Sequence[Any]] = None,
                 labels_path: Optional[str] = None
                ):
        if labels is not None:
            self.labels = labels
        elif labels_path is not None:
            with open(labels_path, "r") as f:
                self.labels = ' '.join(f.readlines()).split()

    @property
    def labels(self) -> List[Any]:
        if not hasattr(self, "_labels"):
            raise ValueError("DataProcessor doesn't set labels or label_mapping yet")
        return self._labels
        
    @labels.setter
    def labels(self, labels: Sequence[Any]):
        if labels is not None:
            self._labels = labels
            self._label_mapping = {k: i for (i, k) in enumerate(labels)}

    @property
    def label_mapping(self) -> Dict[Any, int]:
        if not hasattr(self, "_labels"):
            raise ValueError("DataProcessor doesn't set labels or label_mapping yet")
        return self._label_mapping

    @label_mapping.setter
    def label_mapping(self, label_mapping: Mapping[Any, int]):
        self._labels = [item[0] for item in sorted(label_mapping.items(), key=lambda item: item[1])]
        self._label_mapping = label_mapping
    
    @property
    def id2label(self) -> Dict[int, Any]:
        if not hasattr(self, "_labels"):
            raise ValueError("DataProcessor doesn't set labels or label_mapping yet")
        return {i: k for (i, k) in enumerate(self._labels)}


    def get_label_id(self, label: Any) -> int:
        """get label id of the corresponding label
        Args:
            label: label in dataset
        Returns:
            int: the index of label
        """
        return self.label_mapping[label] if label is not None else None

    def get_labels(self) -> List[Any]:
        """get labels of the dataset
        Returns:
            List[Any]: labels of the dataset
        """
        return self.labels
    
    def get_num_labels(self):
        """get the number of labels in the dataset
        Returns:
            int: number of labels in the dataset
        """
        return len(self.labels)

    def get_train_examples(self, data_dir: Optional[str] = None) -> InputExample:
        """
        get train examples from the training file under :obj:`data_dir`
        call ``get_examples(data_dir, "train")``, see :py:meth:`~openprompt.data_utils.data_processor.DataProcessor.get_examples`
        """
        return self.get_examples(data_dir, "train")

    def get_dev_examples(self, data_dir: Optional[str] = None) -> List[InputExample]:
        """
        get dev examples from the development file under :obj:`data_dir`
        call ``get_examples(data_dir, "dev")``, see :py:meth:`~openprompt.data_utils.data_processor.DataProcessor.get_examples`
        """
        return self.get_examples(data_dir, "dev")

    def get_test_examples(self, data_dir: Optional[str] = None) -> List[InputExample]:
        """
        get test examples from the test file under :obj:`data_dir`
        call ``get_examples(data_dir, "test")``, see :py:meth:`~openprompt.data_utils.data_processor.DataProcessor.get_examples`
        """
        return self.get_examples(data_dir, "test")

    def get_unlabeled_examples(self, data_dir: Optional[str] = None) -> List[InputExample]:
        """
        get unlabeled examples from the unlabeled file under :obj:`data_dir`
        call ``get_examples(data_dir, "unlabeled")``, see :py:meth:`~openprompt.data_utils.data_processor.DataProcessor.get_examples`
        """
        return self.get_examples(data_dir, "unlabeled")

    @abstractmethod
    def get_examples(self, data_dir: Optional[str] = None, split: Optional[str] = None) -> List[InputExample]:
        """get the :obj:`split` of dataset under :obj:`data_dir`
        :obj:`data_dir` is the base path of the dataset, for example:
        training file could be located in ``data_dir/train.txt``
        Args:
            data_dir (str): the base path of the dataset
            split (str): ``train`` / ``dev`` / ``test`` / ``unlabeled``
        Returns:
            List[InputExample]: return a list of :py:class:`~openprompt.data_utils.data_utils.InputExample`
        """
        raise NotImplementedError





class NLPRLProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = None
 
    
    def get_examples(self, data_dir: str, task, level) -> List[InputExample]: 
        img_features = np.loadtxt(f"{data_dir}{task}-{level}/img.csv",dtype=float,delimiter=',')
        txt_inputs = np.loadtxt(f"{data_dir}{task}-{level}/txt.csv",dtype=str,delimiter=',')
        tact_inputs = np.loadtxt(f"{data_dir}{task}-{level}/tact.csv",dtype=int,delimiter=',')
        labels = np.loadtxt(f"{data_dir}{task}-{level}/label.csv",dtype=str,delimiter=',')

        examples = []
        i=0
        for txt, img, tact, label in zip(txt_inputs, img_features, tact_inputs, labels):  
            tact = 1  # to test performance with no tactile signal
            examples.append(InputExample(guid=str(i), text_a=txt , meta = {"img":img, "tact":tact}, tgt_text= label))
            i=i+1
        shuffle(examples)
        return examples
    
class _InputFeatures(InputFeatures):
    
    tensorable_keys = ['img_embeds', 'input_ids', 'inputs_embeds', 'attention_mask', 'token_type_ids', 'label',
        'decoder_input_ids', 'decoder_inputs_embeds', 'soft_token_ids',
        'past_key_values', 'loss_ids']
    all_keys = ['img_embeds','input_ids', 'inputs_embeds', 'attention_mask', 'token_type_ids', 'label',
        'decoder_input_ids', 'decoder_inputs_embeds', 'soft_token_ids',
        'past_key_values', 'loss_ids', 'guid', 'tgt_text', 'encoded_tgt_text', 'input_ids_len']
    non_tensorable_keys = []    
    
    
    def __init__(self,
                img_embeds: Optional=None,
                input_ids: Optional[Union[List, torch.Tensor]] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                attention_mask: Optional[Union[List[int], torch.Tensor]] = None,
                token_type_ids: Optional[Union[List[int], torch.Tensor]] = None,
                label: Optional[Union[int, torch.Tensor]] = None,
                decoder_input_ids: Optional[Union[List, torch.Tensor]] = None,
                decoder_inputs_embeds: Optional[torch.Tensor] = None,
                soft_token_ids: Optional[Union[List, torch.Tensor]] = None,
                past_key_values: Optional[torch.Tensor] = None,  # for prefix_tuning
                loss_ids: Optional[Union[List, torch.Tensor]] = None,
                guid: Optional[str] = None,
                tgt_text: Optional[str] = None,
                use_cache: Optional[bool] = None,
                encoded_tgt_text: Optional[str] = None,
                input_ids_len: Optional[int] = None,
                **kwargs
                ):
        InputFeatures.__init__(self,input_ids, inputs_embeds, attention_mask, token_type_ids, label, decoder_input_ids, decoder_inputs_embeds, soft_token_ids, past_key_values,loss_ids, guid, tgt_text, use_cache, encoded_tgt_text, input_ids_len, **kwargs)
        self.img_embeds = img_embeds
        
        for k in kwargs.keys():
            logger.warning("Your are passing an unexpected key words: {} to InputFeatures, might yield unexpected behaviours!".format(k))
            setattr(self, k, kwargs[k])        

    @classmethod
    def add_tensorable_keys(cls, *args):
        cls.tensorable_keys.extend(args)

    @classmethod
    def add_not_tensorable_keys(cls, *args):
        cls.not_tensorable_keys.extend(args)

    @classmethod
    def add_keys(cls, *args):
        cls.all_keys.extend(args)

    def __repr__(self):
        return str(self.to_json_string())

    def __len__(self):
        return len(self.keys())        
        
    def keys(self, keep_none=False) -> List[str]:
        """get all keys of the InputFeatures
        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.
        Returns:
            :obj:`List[str]`: keys of the InputFeatures
        """
        if keep_none:
            return self.all_keys
        else:
            return [key for key in self.all_keys if getattr(self, key) is not None]  

    def to_dict(self, keep_none=False) -> Dict[str, Any]:
        """get the dict of mapping from keys to values of the InputFeatures
        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.
        Returns:
            :obj:`Dict[str, Any]`: dict of mapping from keys to values of the InputFeatures
        """
        data = {}
        for key in self.all_keys:
            value = getattr(self, key)
            if value is not None:
                data[key] =  value
            elif value is None and keep_none:
                data[key] = None
        return data

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.keys())

    def __setitem__(self, key, item):
        if key not in self.all_keys:
            raise KeyError("Key {} not in predefined set of keys".format(key))
        setattr(self, key, item)

    def values(self, keep_none=False) -> List[Any]:
        """get the values with respect to the keys  of the InputFeatures
        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.
        Returns:
            :obj:`List[Any]`: the values with respect to the keys of the InputFeatures
        """
        return [getattr(self, key) for key in self.keys(keep_none=keep_none)]

    def __contains__(self, key, keep_none=False):
        return key in self.keys(keep_none)

    def items(self,):
        """get the (key, value) pairs  of the InputFeatures
        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.
        Returns:
            :obj:`List[Any]`: the (key, value) pairs of the InputFeatures
        """
        return [(key, self.__getitem__(key)) for key in self.keys()]        
        
    def to_tensor(self, device: str = 'cuda'):
                
        """inplace operation, convert all tensorable features into :obj:`torch.tensor`"""
        for key in self.tensorable_keys:
            value = getattr(self, key)
            if value is not None:
                setattr(self, key, torch.tensor(value))
        return self        
              
    def to(self, device: str = "cuda:0"):
        r"""move the tensor keys to runtime device, such as gpu:0
        """
        for key in self.tensorable_keys:
            value = getattr(self, key)
            if value is not None:
                if key == 'img_embeds' and type(value)==list:  
                    value = torch.stack(value)
                setattr(self, key, value.to(device))
        return self

    def cuda(self):
        r"""mimic the tensor behavior
        """
        return self.to()  
    
    def to_json_string(self, keep_none=False):
        """Serializes this instance to a JSON string."""
        data = {}
        for key in self.all_keys:
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                data[key] =  value.detach().cpu().tolist()
            elif value is None and keep_none:
                data[key] = None
            else:
                data[key] = value
        return json.dumps(data) + "\n"

    
    def collate_fct(batch: List):
        r'''
        This function is used to collate the input_features.
        Args:
            batch (:obj:`List[Union[Dict, InputFeatures]]`): A batch of the current data.
        Returns:
            :obj:`InputFeatures`: Return the :py:class:`~openprompt.data_utils.data_utils.InputFeatures of the current batch of data.
        '''


        elem = batch[0]
        return_dict = {}
        for key in elem:
            if key == "encoded_tgt_text":
                return_dict[key] = [d[key] for d in batch]
            elif key == "img_embeds":
                return_dict[key] = [d[key] for d in batch]   
            elif key == "tgt_text":
                return_dict[key] = [d[key] for d in batch]                 
            else:
                try:
                    return_dict[key] = default_collate([d[key] for d in batch])
                except:
                    pass
                    #print(f"key{key}\n d {[batch[i][key] for i in range(len(batch))]} ")
        return _InputFeatures(**return_dict)
    
    
class PromptDataLoader(object):
    r"""
    PromptDataLoader wraps the original dataset. The input data is firstly wrapped with the
    prompt's template, and then is tokenized by a wrapperd-tokenizer.
    Args:
        dataset (:obj:`Dataset` or :obj:`List`): Either a DatasetObject or a list containing the input examples.
        template (:obj:`Template`): A derived class of of :obj:`Template`
        tokenizer (:obj:`PretrainedTokenizer`): The pretrained tokenizer.
        tokenizer_wrapper_class (:cls:`TokenizerWrapper`): The class of tokenizer wrapper.
        max_seq_length (:obj:`int`, optional): The max sequence length of the input ids. It's used to truncate sentences.
        batch_size (:obj:`int`, optional): The batch_size of data loader
        teacher_forcing (:obj:`bool`, optional): Whether to fill the mask with target text. Set to true in training generation model.
        decoder_max_length (:obj:`int`, optional): the decoder maximum length of an encoder-decoder model.
        predict_eos_token (:obj:`bool`, optional): Whether to predict the <eos> token. Suggest to set to true in generation.
        truncate_method (:obj:`bool`, optional): the truncate method to use. select from `head`, `tail`, `balanced`.
        kwargs  :Other kwargs that might be passed into a tokenizer wrapper.
    """
    def __init__(self,
                 dataset: Union[Dataset, List],
                 template: Template,
                 tokenizer_wrapper: Optional[TokenizerWrapper] = None,
                 tokenizer: PreTrainedTokenizer = None,
                 tokenizer_wrapper_class = None,
                 verbalizer: Optional[Verbalizer] = None,
                 max_seq_length: Optional[str] = 512,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = False,
                 teacher_forcing: Optional[bool] = False,
                 decoder_max_length: Optional[int] = -1,
                 predict_eos_token: Optional[bool] = False,
                 truncate_method: Optional[str] = "tail",
                 drop_last: Optional[bool] = False,
                 **kwargs,
                ):

        assert hasattr(dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {dataset}"
        assert hasattr(dataset, "__len__"), f"The dataset must have __len__ method. dataset is {dataset}"
        self.raw_dataset = dataset

        self.wrapped_dataset = []
        self.tensor_dataset = []
        self.template = template
        self.verbalizer = verbalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.teacher_forcing = teacher_forcing

        if tokenizer_wrapper is None:
            if tokenizer_wrapper_class is None:
                raise RuntimeError("Either wrapped_tokenizer or tokenizer_wrapper_class should be specified.")
            if tokenizer is None:
                raise RuntimeError("No tokenizer specified to instantiate tokenizer_wrapper.")

            tokenizer_wrapper_init_keys = signature(tokenizer_wrapper_class.__init__).args
            prepare_kwargs = {
                "max_seq_length" : max_seq_length,
                "truncate_method" : truncate_method,
                "decoder_max_length" : decoder_max_length,
                "predict_eos_token" : predict_eos_token,
                "tokenizer" : tokenizer,
                **kwargs,
            }

            to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}
            self.tokenizer_wrapper = tokenizer_wrapper_class(**to_pass_kwargs)
        else:
            self.tokenizer_wrapper = tokenizer_wrapper

        # check the satisfiability of each component
        assert hasattr(self.template, 'wrap_one_example'), "Your prompt has no function variable \
                                                         named wrap_one_example"

        # process
        self.wrap()
        self.tokenize()

        if self.shuffle:
            sampler = RandomSampler(self.tensor_dataset)
        else:
            sampler = None
        
        self.dataloader = DataLoader(
            self.tensor_dataset,
            batch_size = self.batch_size,
            sampler= sampler,
            collate_fn = _InputFeatures.collate_fct,
            drop_last = drop_last,
        )


    def wrap(self):
        r"""A simple interface to pass the examples to prompt, and wrap the text with template.
        """
        if isinstance(self.raw_dataset, Dataset) or isinstance(self.raw_dataset, List):
            assert len(self.raw_dataset) > 0, 'The dataset to be wrapped is empty.'
            # for idx, example in tqdm(enumerate(self.raw_dataset),desc='Wrapping'):
            for idx, example in enumerate(self.raw_dataset):
                if self.verbalizer is not None and hasattr(self.verbalizer, 'wrap_one_example'): # some verbalizer may also process the example.
                    example = self.verbalizer.wrap_one_example(example)
                wrapped_example = self.template.wrap_one_example(example)
                self.wrapped_dataset.append(wrapped_example)
        else:
            raise NotImplementedError

    def tokenize(self) -> None:
        r"""Pass the wrapped text into a prompt-specialized tokenizer,
           the true PretrainedTokenizer inside the tokenizer is flexible, e.g. AlBert, Bert, T5,...
        """
        for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset),desc='tokenizing'):
        # for idx, wrapped_example in enumerate(self.wrapped_dataset):
            inputfeatures = _InputFeatures(**self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing), **wrapped_example[1]).to_tensor()
            self.tensor_dataset.append(inputfeatures)

    def __len__(self):
        return  len(self.dataloader)

    def __iter__(self,):
        return self.dataloader.__iter__()




