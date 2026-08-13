from torch.utils.data import Dataset 
class set_(Dataset):
    def __init__(this, X, y):
        this.x_i = X; this.y_i = y
    def __len__(this):
        return len(this.x_i)
    def __getitem__(this, i):
        return this.x_i[i], this.y_i[i]

