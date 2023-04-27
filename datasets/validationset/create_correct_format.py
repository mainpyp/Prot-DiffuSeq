import json
import os
import tqdm


def create_correct_file(fasta_seq: str, fasta_struc: str) -> None:
    """ This function takes two fasta files with the same headers as input. 
    fasta_seq: fasta file with the amino acid sequence
    fasta_struc: fasta file with the structure sequence
    
    The function creates a json file with the following format:
    Each sequence is in one line in in that json structure:
    {"src": "structure", "trg": "sequence"}
    """
    
    output_path = "/".join(fasta_seq.split("/")[:-1]) + "/"
    output_filename = fasta_seq.split("/")[-1].replace(".fasta", ".jsonl")
    output_filename = output_path + output_filename
    if os.path.exists(output_filename):
        os.remove(output_filename)

    print(output_filename)

    processed_lines = []
    with open(fasta_seq) as f_seq, open(fasta_struc) as f_struc:
        for index, (seq_line, struc_line) in tqdm.tqdm(enumerate(zip(f_seq, f_struc))):
            if index == 0 or seq_line.startswith(">") or struc_line.startswith(">"):
                continue

            seq = seq_line.strip()
            struc = struc_line.strip()



            assert len(seq) == len(struc), f"Missing AA or structure token for {seq} in line {index}"

            write_string = f'{{"src": "{struc}", "trg": "{seq}"}}\n'
            processed_lines.append(write_string)
    print("Done processing file")

    print(f"Write file {output_filename}")
    with open(output_filename, "a") as of:
        of.writelines(processed_lines)

if __name__ == '__main__':
    create_correct_file("/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/datasets/validationset/val_AA.fasta", 
                        "/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/datasets/validationset/val_SS.fasta")