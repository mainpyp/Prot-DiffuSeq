import argparse
import itertools
import os
import tqdm


parser = argparse.ArgumentParser(description='Put into correct format.')
parser.add_argument('--input_path', required=True, help='The input path for the datafile')

args = parser.parse_args()

def create_file(filename: str) -> None:
    output_filename = filename.split("/")[-1].replace(".csv", ".jsonl")
    if os.path.exists(output_filename):
        os.remove(output_filename)


    print(output_filename)
    with open(filename) as f:
        f.readline()  # skip header

        for index, (structure_string, seq_string) in tqdm.tqdm(enumerate(itertools.zip_longest(*[f]*2))):
            if index == 0 or structure_string == None or seq_string == None:
                continue

            sequence = " ".join(list(seq_string.split(',')[-1].strip()))
            structure =  " ".join(list(structure_string.split(',')[-1].strip()))

            assert len(sequence) == len(structure), f"Missing AA or structure token for {sequence} in line {index}"

            write_string = f'{{"src": "{structure}", "trg": "{sequence}"}}'

            with open(output_filename, "a") as of:
                of.write(write_string + "\n")


if __name__ == '__main__':
    create_file(args.input_path)