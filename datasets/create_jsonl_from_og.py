import argparse
import itertools
import os

import pandas as pd
import tqdm


parser = argparse.ArgumentParser(description='Put into correct format.')
parser.add_argument('--input_path', required=True, help='The input path for the datafile')

args = parser.parse_args()


def create_file(filename: str) -> None:
    output_filename = filename.split("/")[-1].replace(".csv", ".jsonl")
    if os.path.exists(output_filename):
        os.remove(output_filename)

    print(output_filename)

    processed_lines = []
    with open(filename) as f:
        f.readline()  # skip header

        for index, (structure_string, seq_string) in tqdm.tqdm(enumerate(itertools.zip_longest(*[f]*2))):
            if structure_string == None or seq_string == None:
                continue
            
            structure_items = structure_string.split(",")
            sequence = " ".join(list(seq_string.split(",")[-1].strip()))
            structure =  " ".join(list(structure_items[-1].strip()))
            af_id = structure_items[1]

            assert len(sequence) == len(structure), f"Missing AA or structure token for {sequence} in line {index}"

            write_string = f'{{"src": "{sequence}", "trg": "{structure}", "af_id": {af_id}}}\n'
            processed_lines.append(write_string)
            
            # empty the list every 1M lines and write to file
            if index % 1_000_000 == 0:
                print(f"Write in file index {index}")
                with open(output_filename, "a") as of:
                    of.writelines(processed_lines)
                processed_lines = []
            
        print("Last append")
        with open(output_filename, "a") as of:
                of.writelines(processed_lines)
        print("Done processing file")


if __name__ == '__main__':
    import time
    t0 = time.time()
    create_file(args.input_path)
    d = time.time() - t0
    print("duration: %.2f s." % d)
    