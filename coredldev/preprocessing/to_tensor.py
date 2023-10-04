import torch


def to_tensor(device="cpu"):
    def inner(a, *args):
        for i in a:
            if torch.is_tensor(i):
                i.to(device=device)
            else:
                i = torch.tensor(i).to(device=device)
        return list(a)

    return inner

def to_tensor_clean(device="cpu", debug = False):
    def inner(a):
        a["params"]["ra"] = a["params"]["angle"][0]
        a["params"]["dec"] = a["params"]["angle"][1]
        a["params"]["psi"] = a["params"]["angle"][2]
        del a["params"]["angle"]
        if debug == True:
            print(a["params"])
        return torch.tensor(a["signal"]).to(device), torch.tensor(list(a["params"].values())).to(device),
    return inner