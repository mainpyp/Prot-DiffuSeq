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


        processed_lines = []

        seq = ""
        struc = ""
        for index, line in enumerate(f):
            splitted = line.split(",")
            if len(splitted) <= 1: continue
            
            id = splitted[1]
            if index % 2 == 0:
                seq = " ".join(list(splitted[-1].strip()))
            else:
                struc = " ".join(list(splitted[-1].strip()))
            
            if struc != "":
                assert len(seq) == len(struc), f"Missing AA or structure token for {seq} in line {index}"
                write_string = f'{{"src": "{struc}", "trg": "{seq}", "af_id": "{id}"}}\n'
                processed_lines.append(write_string)
                seq = struc = ""                
            
            # empty the list every 1M lines and write to file
            if index % 500_000 == 0:
                print(f" -> Write in file index {index}")
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
    