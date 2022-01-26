from dataloader import DataLoader


class DataSettings:
    debug = False
    batch_size = 64
    ncond = 20


DataLoader(None, opt=DataSettings, dataset="i80")
