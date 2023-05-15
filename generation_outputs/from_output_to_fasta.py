import json
from dataset_utils import remove_brackets


def convert_recs(path: str) -> None:
    """ Converts a jsonl file to a fasta file. Each sequence is in the source file 
    under the key recovery and for each sequence a 10 character unique key is generated 
    together with a trailig index that is spereated with a "," which is used as 
    a header for the fasta file format.
    """
    with open(path, "r") as f:
        jsonl = [json.loads(line) for line in f]
    with open(path.replace(".json", "_rec.fasta"), "w") as f:
        for i, seq in enumerate(jsonl):
            f.write(f">{seq['af_id']}\n{remove_brackets(seq['recover'])}\n")

def convert_refs(path: str) -> None:
    """ Converts a jsonl file to a fasta file. Each sequence is in the source file 
    under the key recovery and for each sequence a 10 character unique key is generated 
    together with a trailig index that is spereated with a "," which is used as 
    a header for the fasta file format.
    """
    with open(path, "r") as f:
        jsonl = [json.loads(line) for line in f]
    with open(path.replace(".json", "_ref.fasta"), "w") as f:
        for i, seq in enumerate(jsonl):
            f.write(f">{seq['af_id']}\n{remove_brackets(seq['reference'])}\n")
    
    

if __name__ == "__main__":
    convert_recs("/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/diffuseq_ProtMediumCorrect_h256_lr1e-05_t6000_sqrt_lossaware_seed123_pm-correct-new-params20230419-17:39:32/correctones/ema_0.9999_100000.pt.samples/seed123_step0.json")
    convert_refs("/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/diffuseq_ProtMediumCorrect_h256_lr1e-05_t6000_sqrt_lossaware_seed123_pm-correct-new-params20230419-17:39:32/correctones/ema_0.9999_100000.pt.samples/seed123_step0.json")
    convert_recs("/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/diffuseq_ProtMediumCorrect_h256_lr1e-05_t6000_sqrt_lossaware_seed123_pm-correct-new-params20230419-17:39:32/correctones/ema_0.9999_130000.pt.samples/seed123_step0.json")
    convert_refs("/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/diffuseq_ProtMediumCorrect_h256_lr1e-05_t6000_sqrt_lossaware_seed123_pm-correct-new-params20230419-17:39:32/correctones/ema_0.9999_130000.pt.samples/seed123_step0.json")
