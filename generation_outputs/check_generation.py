import json

def load_jsonl_file(path: str):
    """
    Load a jsonl file.
    """
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]
    
if __name__ == '__main__':
   
    data = load_jsonl_file("generation_outputs/diffuseq_ProtMediumCorrect_h128_lr0.001_t4000_sqrt_lossaware_seed102_MediumDatasetCorrectScope20230323-11:29:21/ema_0.9999_002000.pt.samples/seed123_step0.json")
    print(data[0])