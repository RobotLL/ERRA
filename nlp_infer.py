# pip install openprompt
import os
import json
import clip
import torch
import numpy as np
from openprompt.plms import load_plm
from openprompt import PromptForGeneration
from NLP.template import PrefixTuningTemplate
from NLP.data_util import _InputFeatures, PromptDataLoader
from openprompt.data_utils import InputExample
from openprompt.utils import signature
from PIL import Image

# high_level_collect_number = 1
# language_input_list1, image_input_list1, tactile_input_list1, label_list1, = drawer_collect(
#     high_level_collect_number)

# device = "cpu"
# prompt_path = './NLP/prompt_model'


class Planner:
    def __init__(self,
                 prompt_path: str = './NLP/prompt_model',
                 device: str = None,
                 ):
        self.device = device
        # load nlp model
        config = json.load(open(os.path.join(prompt_path, 'config.json'), 'r', encoding='utf-8'))
        plm, self.tokenizer, model_config, self.WrapperClass = load_plm(
            config['plm_model'], config['plm_model_name_or_path'])

        self.mytemplate = PrefixTuningTemplate(
            model=plm, tokenizer=self.tokenizer, text=config['text'],  num_token=config['num_prefix'], using_decoder_past_key_values=config['using_decoder_past_key_values'])
        nlp_model = PromptForGeneration(
            plm=plm, template=self.mytemplate, freeze_plm=True, tokenizer=self.tokenizer, plm_eval_mode=True)
        nlp_model.template.load_state_dict(torch.load(os.path.join(
            prompt_path, 'prompt_parameter.pt'), map_location=torch.device('cpu')))
        self.nlp_model = nlp_model.to(self.device)
        self.nlp_model.eval()

        # load clip
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def Image2embedding(self, image):
        with torch.no_grad():

            image = self.preprocess(Image.open(image)).unsqueeze(0).to(self.device)
            # image = self.preprocess(image).unsqueeze(0).to(self.device)
            img_feature = self.clip_model.encode_image(
                image).squeeze().cpu().numpy()  # size [1,512]

        return img_feature

    def test_nlp_infer(self, image, sentence, tactile):
        if type(image) == str:
            img_feature = self.Image2embedding(image)
        else:
            img_feature = image
        input_example = [InputExample(guid='0', text_a=sentence, meta={
                                      "img": img_feature, "tact": tactile}, tgt_text='')]
        dataloader = PromptDataLoader(dataset=input_example, template=self.mytemplate, tokenizer=self.tokenizer,
                                      tokenizer_wrapper_class=self.WrapperClass, max_seq_length=256, decoder_max_length=256,
                                      batch_size=1, shuffle=False, teacher_forcing=False, predict_eos_token=True,
                                      truncate_method="head")

        generation_arguments = {
            "max_length": 20,
            "max_new_tokens": None,
            "min_length": 1,
            "temperature": 1.0,
            "do_sample": False,
            "top_k": 0,
            "top_p": 0.9,
            "repetition_penalty": 1.0,  # 1.0 means no penalty
            "num_return_sequences": 1,
            "num_beams": 1,
            "bad_words_ids": [[628], [198]]
        }

        for inputs in dataloader:
            inputs = inputs.to(self.device)
            output_ids, output_sentence = self.nlp_model.generate(inputs, **generation_arguments)
        output_id = sum(output_ids[0])
        return output_id, output_sentence[0]


# def main():
#     planner = Planner(prompt_path, device)
#     succ = 0
#     for i in range(high_level_collect_number):
#         image = image_input_list1[i]
#         sentence = language_input_list1[i]
#         tactile = tactile_input_list1[i]
#         label = label_list1[i]
#         output_id, output_sentence = planner.test_nlp_infer(image, sentence, tactile)
#         if output_sentence == label:
#             succ += 1
#         print(output_sentence)
#         print(label)

#     print('succ_rate', succ/high_level_collect_number)


# if __name__ == "__main__":
#     main()


# todo2:
# save Test each sentence in each task, and save the non repeated output separately
