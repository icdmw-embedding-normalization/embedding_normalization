class PerformanceLogger:
    def __init__(self):
        self.min_valid_loss = float("inf")
        self.min_valid_epoch = -1
        self.min_valid_train_loss = float("inf")
        self.min_valid_test_loss = float("inf")
        self.min_valid_valid_acc = 0
        self.min_valid_test_acc = 0

        self.min_test_loss = float("inf")
        self.min_test_epoch = -1
        self.min_test_train_loss = float("inf")
        self.min_test_valid_loss = float("inf")
        self.min_test_valid_acc = 0
        self.min_test_test_acc = 0

    def update(self, epoch_idx, train_loss, valid_loss, test_loss, valid_acc=None, test_acc=None):
        if self.min_valid_loss > valid_loss:
            self.min_valid_loss = valid_loss
            self.min_valid_epoch = epoch_idx
            self.min_valid_train_loss = train_loss
            self.min_valid_test_loss = test_loss
            self.min_valid_valid_acc = valid_acc
            self.min_valid_test_acc = test_acc

        if self.min_test_loss > test_loss:
            self.min_test_loss = test_loss
            self.min_test_epoch = epoch_idx
            self.min_test_train_loss = train_loss
            self.min_test_valid_loss = valid_loss
            self.min_test_valid_acc = valid_acc
            self.min_test_test_acc = test_acc

        line = (
            f"{epoch_idx} {train_loss:.6f} {valid_loss:.6f} {test_loss:.6f}"
            f" | {self.min_valid_epoch} {self.min_valid_loss:.6f} {self.min_valid_test_loss:.6f}"
            f" | {self.min_test_epoch} {self.min_test_loss:.6f}"
        )
        if test_acc is not None and valid_acc is not None:
            line += f" | {valid_acc:.6f} {test_acc:.6f}"
            line += f" | {self.min_valid_valid_acc:.6f} {self.min_valid_test_acc:.6f} {self.min_test_valid_acc:.6f} {self.min_test_test_acc:.6f}"

        print(line)
