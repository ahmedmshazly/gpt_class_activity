import torch
import torch
class ModelTrainer:
    """
    Handles the training and evaluation of a GPT language model.
    """

    def __init__(self, model, optimizer, data_loader, device):
        """
        Initializes the ModelTrainer.

        Args:
            model (nn.Module): The GPT language model to train.
            optimizer (torch.optim.Optimizer): Optimizer for training the model.
            data_loader (DataLoader): DataLoader instance for fetching data.
            device (torch.device): Device on which to train the model.
        """
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.device = device

    @torch.no_grad()
    def estimate_loss(self, eval_iters):
        """
        Estimates the loss of the model on the training and validation datasets.

        Args:
            eval_iters (int): Number of iterations to use for loss estimation.

        Returns:
            dict: A dictionary containing average loss for 'train' and 'val' splits.
        """
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.data_loader.get_batch(split, self.device)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self, max_iters, eval_iters):
        """
        Trains the GPT language model.

        Args:
            max_iters (int): Maximum number of iterations for training.
            eval_iters (int): Interval for evaluating the model.
        """
        for iter in range(max_iters):
            print(f"Iteration: {iter}")
            if iter % eval_iters == 0:
                losses = self.estimate_loss(eval_iters)
                print(f"Step: {iter}, Train Loss: {losses['train']:.3f}, Val Loss: {losses['val']:.3f}")

            xb, yb = self.data_loader.get_batch('train', self.device)

            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            print(f"Iteration {iter} Loss: {loss.item()}")

    def save_model(self, filename):
        """
        Saves the model to a file.

        Args:
            filename (str): The filename to save the model to.
        """
        with open(filename, 'wb') as f:
            torch.save(self.model.state_dict(), f)
        print('Model saved to', filename)

    def load_model(self, filename):
        """
        Loads the model from a file.

        Args:
            filename (str): The filename to load the model from.
        """
        with open(filename, 'rb') as f:
            self.model.load_state_dict(torch.load(f))
        print('Model loaded from', filename)




