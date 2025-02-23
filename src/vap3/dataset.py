from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
from typing import List


class Dataset(Dataset):
    def __init__(self, dataPath, tokenizer, type,ds,mode):
        self.dataPath = dataPath
        self.type = type
        self.tokenizer = tokenizer
        self._pad_values = {
            "input_ids": self.tokenizer.pad_token_id,
            "attention_mask": 0,
            "token_type_ids": self.tokenizer.pad_token_type_id,
            "special_tokens_mask": 1,
        }
        self.ds = ds
        self.mode=mode
        self.read()
        
    def __getitem__(self,index):
        keys = [*self.MAPScore]
        queryID = keys[index]
        queryText = self.queryDic[queryID]

        output = self.tokenizer.encode_plus(
                queryText,
                add_special_tokens=True,
                return_tensors="pt",)
        assert all(v.size(0) == 1 for v in output.values())


        return output , queryID, [self.MAPScore[queryID]]

    def __len__(self):
        return (len(self.MAPScore))

    def read(self):
        import csv
        self.queryDic = {}
        self.MAPScore = {}

        query_file = f'/home/narabzad/prompteng/data/{self.ds}/llama3_{self.type}_{self.ds}_{self.mode}.performance.jsonl.tsv'


        for row in open(query_file,'r').readlines():
            id = (row.split('\t')[0])
            self.queryDic[(id)]= row.split('\t')[1]


        MAPScore_file = f'/home/narabzad/prompteng/data/{self.ds}/llama3_{self.type}_{self.ds}_{self.mode}.performance.jsonl.tsv'

        read_tsv = csv.reader(MAPScore_file, delimiter="\t")
        c=0
        for row in open(MAPScore_file,'r').readlines():
            id = (row.split('\t')[0])

            c+=1
            #self.MAPScore[int(row[0])] = float(row[2])
            #if row[0].isdigit():
            self.MAPScore[id] = float(row.split('\t')[2])
            #else:
            #    print(row)
        print()