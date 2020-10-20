class EarlyStopper:
    IMPROVED = "IMPROVED"
    WORSEND = "WORSEND"
    FINISH = "FINISH"

    def __init__(self, early_stop_epochs):
        self.early_stop_epochs = early_stop_epochs
        self.min_valid_loss = float("inf")
        self.tolerance_epochs = 0

    def step(self, valid_loss):
        if self.min_valid_loss < valid_loss:
            self.tolerance_epochs += 1

            if self.tolerance_epochs >= self.early_stop_epochs:
                return EarlyStopper.FINISH
            else:
                return EarlyStopper.WORSEND

        else:
            self.min_valid_loss = valid_loss
            self.tolerance_epochs = 0
            return EarlyStopper.IMPROVED
