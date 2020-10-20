class IterLogger:
    def __init__(self, model_name, dataset_name, label, sparse_embedding, distortion="none"):
        filename = f"iter-{label}-{model_name}-sparse_embedding={sparse_embedding}-{dataset_name}"
        if distortion != "none":
            filename += f"-{distortion}"
        self.f = open(f"{filename}.txt", "w")

    def write(self, total_iter, data, print_std=False):
        line = "\t".join([str(datum) for datum in data])
        if print_std:
            print(line)
        self.f.write(f"{total_iter}\t{line}\n")

    def close(self):
        self.f.close()
