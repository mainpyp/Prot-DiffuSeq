


with open("/Users/adrianhenkel/Documents/Programming/git/github/Prot-DiffuSeq/generation_outputs/test_sequences/semi_random_generation/semi_random_AA.fasta") as file:
    max = 0
    for line in file:
        max = len(line) if len(line) > max else max
    print(max)