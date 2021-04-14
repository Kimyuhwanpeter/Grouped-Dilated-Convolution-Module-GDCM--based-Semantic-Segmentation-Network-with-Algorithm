import os, sys


def main(parser, path):
    from datasets import DataManager
    datamanager = DataManager(**parser.dataset)
    loader_test = datamanager.loader_test

    import torch
    from models import CenterNet, dlaseg
    # model = dlaseg(num_classes=loader_train.dataset.num_classes,
    #                **parser.model)
    model = CenterNet(dlaseg(pretrained_base=False),
                      head_conv=128,
                      abnormal=True)
    model.load_state_dict(torch.load(path)['state_dict'])

    from engine import build_engine
    engine_ = build_engine(model, None, None, None, None, **parser.engine)

    engine_.evaluate(loader_test, None)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("not matched argument")

    sys.path.append(os.path.dirname(__file__))

    from common import Parser
    parser = Parser(sys.argv[1], False)

    main(parser, sys.argv[2])
