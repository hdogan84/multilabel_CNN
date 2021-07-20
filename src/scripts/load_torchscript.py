import torch

if __name__ == "__main__":
    # Ask user an create build folder with contents

    model = torch.jit.load("model.pt", map_location=torch.device("cuda"))
    result = model(torch.rand(64, 1, 128, 256).to("cuda"))
    print(result)
    print(result.size())
