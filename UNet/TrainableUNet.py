import torch
import UNet

class TrainableUNet(UNet):

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_predicted = self.forward(x)
        assert y_predicted.shape == y.shape
        loss = ((y_predicted - y) ** 2).mean() 
        if self.train:
            self.log("train_loss", loss, prog_bar=True)
            if(batch_idx==500):
              torch.save(model.state_dict(), '/content/drive/My Drive/CCA-Projekt/Data/model_run')
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)