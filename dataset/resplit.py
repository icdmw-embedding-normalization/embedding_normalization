import random
import os


def main(dataset_name, sample_rate):
    seed = 1
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    lines = []
    for filename in ["train", "valid", "test"]:
        with open(f"{dataset_name}/{filename}.libsvm", "r") as f:
            for line in f:
                if random.random() < sample_rate:
                    lines.append(line)

    print(lines[0])
    random.shuffle(lines)
    print(lines[0])
    print(len(lines))

    n_train = int(len(lines) * 0.8)
    n_valid = int(len(lines) * 0.9)

    train_lines = lines[:n_train]
    valid_lines = lines[n_train:n_valid]
    test_lines = lines[n_valid:]

    for key, lines in [
        ("train", train_lines),
        ("valid", valid_lines),
        ("test", test_lines),
    ]:
        with open(f"{dataset_name}/{key}-{seed}.libsvm", "w") as f:
            for line in lines:
                f.write(line)


if __name__ == "__main__":
    main("avazu", sample_rate=0.1)
