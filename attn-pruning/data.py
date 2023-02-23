import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HL_TOKEN = '[HL]'
MAX_QUESTION_LENGTH = 32
MAX_CONTEXT_LENGTH = 480
MAX_INPUT_LENGTH = MAX_CONTEXT_LENGTH + MAX_QUESTION_LENGTH # max:1024
DEVICE = torch.device('mps')

tokenizer = AutoTokenizer.from_pretrained("p208p2002/bart-squad-qg-hl")
model = AutoModelForSeq2SeqLM.from_pretrained("p208p2002/bart-squad-qg-hl")

class SquadQGDataset(Dataset):
    def __init__(self, split_set:str='train', tokenizer=tokenizer, is_test=False):
        """
        Args:
            split_set(str): `train` or `validation`
            tokenizer(transformers.PreTrainedTokenizer)
        """
        dataset = load_dataset("squad")
        self.split_set = split_set
        self.is_test = is_test
        self.data = dataset[split_set]
        self.tokenizer = tokenizer
        
    def __getitem__(self,index):
        data = self.data[index]
        # print(data['context'])
        answer_text = data['answers']['text'][0]
        answer_len = len(answer_text)
        answer_start = data['answers']['answer_start'][0] 
        hl_context = data['context'][:answer_start] + HL_TOKEN + answer_text + HL_TOKEN + data['context'][answer_start + answer_len:]

        if self.is_test == False:
            model_input = self.prepare_input(context=hl_context + self.tokenizer.sep_token,label=data['question'] + self.tokenizer.eos_token)
            #return model_input['input_ids'],model_input['labels'] 
        else:
            model_input = self.prepare_input(context=hl_context + self.tokenizer.sep_token)
            #return model_input['input_ids'],data['question']
        return model_input # // model needs a dictionary
        
    def __len__(self):
        return len(self.data)

    def convert_to_tensor(self,model_input):
        for key in model_input.keys():
            model_input[key] = torch.LongTensor(model_input[key])
        return model_input

    def prepare_input(self,context,label=None):
        tokenizer = self.tokenizer
        pad_token_id = self.tokenizer.pad_token_id

        if label is None:
            model_input = tokenizer(context,max_length=MAX_CONTEXT_LENGTH,truncation=True)
            return self.convert_to_tensor(model_input)

        context_input = tokenizer(context)
        label_input = tokenizer(label)
        
        # limit context length
        model_input = {}
        model_input['input_ids'] = context_input['input_ids'][:MAX_CONTEXT_LENGTH] + label_input['input_ids'][:MAX_QUESTION_LENGTH]
        
        # prepars lables
        model_input['labels'] = model_input['input_ids'][:]
        for i,_ in enumerate(context_input['input_ids'][:MAX_CONTEXT_LENGTH]):
            model_input['labels'][i] = -100 # set the context part to -100 for ignore loss

        # pad or limit to max length
        pad_ids = [pad_token_id]*MAX_INPUT_LENGTH
        pad_labels = [-100]*MAX_INPUT_LENGTH
        model_input['input_ids'] = (model_input['input_ids'] + pad_ids)[:MAX_INPUT_LENGTH] 
        model_input['labels'] = (model_input['labels'] + pad_labels)[:MAX_INPUT_LENGTH]        

        model_input["attention_mask"] = [i == pad_token_id for i in model_input['input_ids']]
        # // required key in model input

        return self.convert_to_tensor(model_input)

class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 1

        self.train_dataset = SquadQGDataset(split_set='train')
        self.dev_dataset = SquadQGDataset(split_set='validation')
        self.test_dataset = SquadQGDataset(split_set='validation',is_test=True)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)