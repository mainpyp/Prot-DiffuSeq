NLINES = 300
PATH = "datasets/ProtMinimalStrucToSeq/"

# Taken from sample AF-A0A4Q2Y815-F1
SEQUENCE = "MAETWQRLNPGARVHIAYGGTRERFAKLCWPDRTFIDDPRLRTRDHQRDRQSYLGLMRAVVAELQPLRMDRVLMAECDVMPLRRGLVDYLEKRRCEEGADLLGPRIRRVDGTGHPHYLAHQFHPAFGQWLAQSVRAEKETVLMMVGCLTWWTWKAFEAVSASAEPMPVYLELAIPTAAHQLGFRVRELTELNGFLQPLGEMQPHLAEWRAGGHWVAHPCKKIWRRASPS"
STRUCTURE = "dqvqvcvqevvdddaaedaddpvvvvpdpddhydyqpdclcvdpdllvsqraclsvllrvcvscvvvpdfkdkdadpqkdfqhhclvvvvvvvcvvqvfqkeaapkdwcpphppplcvvpvpppvvlvclvlapdppsrtwifgpdrimmhgpvlsvqlspdddsdrgrhrgsnqsssvsppghydyphlchvqadpddacvvcsvvcsvvvhgihprhvpppvppddd"
SEQUENCE = " ".join(list(SEQUENCE))
STRUCTURE = " ".join(list(STRUCTURE))

def create_train_test_val_files(path: str):
    """ This function creates three files (train.jsonl, test.jsonl, valid.jsonl) in the given path. 
    Each file contains 500 lines in the format {"trg": SEQUENCE, "src": STRUCTURE}.
    """
    with open(path + 'train.jsonl', 'w') as f:
        for i in range(NLINES):
            f.write(f'{{"src":"{STRUCTURE}", "trg":"{SEQUENCE}"}}\n')
    
    print("Done creating train.jsonl file")

    with open(path + 'test.jsonl', 'w') as f:
        for i in range(NLINES):
            f.write(f'{{"src":"{STRUCTURE}", "trg":"{SEQUENCE}"}}\n')
    print("Done creating test.jsonl file")

    with open(path + 'valid.jsonl', 'w') as f:
        for i in range(NLINES):
            f.write(f'{{"src":"{STRUCTURE}", "trg":"{SEQUENCE}"}}\n')
    print("Done creating valid.jsonl file")

if __name__ == "__main__":
    create_train_test_val_files(PATH)