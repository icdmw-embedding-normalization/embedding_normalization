from pathlib import Path


def save(line, filename="unknown.txt"):
    dir_path = Path("./results")
    dir_path.mkdir(exist_ok=True)

    with open(dir_path / filename, "a") as f:
        print(line)
        f.write(line + "\n")
        f.close()
