from n_body.dataloader import *

if __name__ == "__main__":

    train, val, test = get_nbody_dataloaders()

    for i, (loc, vel, edges, charges) in enumerate(train):
        print(f"{i}: {loc}")
