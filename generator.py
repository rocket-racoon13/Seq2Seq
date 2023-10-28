import os
from tqdm import tqdm
from typing import List

import torch


class Generator:
    def __init__(
        self,
        args,
        tokenizer,
        model,
        device
    ):
        self.args = args
        self.data_dir = args.inference_data_dir
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
    
    def generate(self):
        inputs = self.preprocess_data()
        results = self.generate_next_token(inputs)
        self.save_results(results)
    
    def save_results(self, results: List[str]):
        with open(
            os.path.join(self.args.save_dir, "inference_result.txt"), "w",
            encoding="utf-8-sig") as f_out:
            f_out.write("\n".join(results))
        print(f"=== Inference result saved to -> {self.args.save_dir}/inference_result.txt ===")
    
    def generate_next_token(self, inputs):
        results = []
        with torch.no_grad():
            inputs = [input.to(self.device) for input in inputs]
            for input in tqdm(inputs):
                y_pred = self.model(input.unsqueeze(dim=0))[:,-1,:]
                pred_idx = torch.argmax(y_pred.squeeze())
                pred_token = self.tokenizer.decode(pred_idx.detach().cpu().item())
                results.append(pred_token)
        return results
                
    def preprocess_data(self):
        # Load inference data
        if not os.path.exists(self.data_dir):
            raise IOError("Data for inference do not exist.")
        else:
            with open(self.data_dir, encoding="utf-8-sig") as f_in:
                data = f_in.read().splitlines()

        # Apply encoding to inputs and tensorize
        inputs = [self.tokenizer.encode_sequence(sent.split()) for sent in data]
        inputs = [torch.LongTensor(input) for input in inputs]
        return inputs