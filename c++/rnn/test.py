import torch

if __name__ == '__main__':
    

    # input from 0 to 9
    

    x = [[0.5, 1, ],
          [1, 1, ],
          [0.3, 1, ]]

    x1 = [[0.6, 2, ],
          [2, 2, ],
          [0.2, 2, ]]

    
    

    y = torch.tensor(x, dtype=torch.float32).view(3, 2)
    y1 = torch.tensor(x1, dtype=torch.float32).view(3, 2)

    arr = [y, y1]
    print (arr)
    print (torch.stack(arr, 1))
