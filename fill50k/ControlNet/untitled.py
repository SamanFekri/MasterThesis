class MyCLDM(ControlLDM):
    
    def validation_step():
        img = batch
        wandb log (img)
        super.validation_step()