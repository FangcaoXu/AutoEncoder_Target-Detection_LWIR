import torch
import numpy as np
from train import load_model_predict
from csv2npz import parse_csv


def main(args):
    cuda = not args.no_cuda and torch.cuda.is_available()

    with torch.no_grad():
        model, x_norm, y_norm = load_model_predict(args.model, cuda)
        # myString = ",".join(myList) change a list to string
        #
        X = np.clip(parse_csv(open(args.input), header=1), 0, x_norm) / x_norm
        X = torch.from_numpy(np.expand_dims(np.expand_dims(X, 0), 0)).float()

        if cuda:
            X = X.cuda()

        Y, _ = model(X)

        if cuda:
            Y = Y.cpu()

        arr = np.squeeze(Y.data.numpy()) * y_norm
        print(arr.shape)
        np.savetxt(args.output, arr, delimiter=",")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CAE Encoder')

    # data options
    parser.add_argument('--input', type=str, required=True, help='input csv files, separated by comma')
    parser.add_argument('--model', type=str, required=True, help='model file')
    parser.add_argument('--output', type=str, required=True, help='ouput csv file')

    # settings
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disable CUDA')
    main(parser.parse_args())
