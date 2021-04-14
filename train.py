import os, sys


def main(parser, logger):
    from datasets import DataManager
    datamanager = DataManager(**parser.dataset)
    loader_train = datamanager.loader_train
    loader_test = datamanager.loader_test

    from models import CenterNet, dlaseg
    model = dlaseg(num_classes=loader_train.dataset.num_classes,
                   **parser.model)
    # model = CenterNet(dlaseg(), head_conv=128, abnormal=True)

    from modules import build_optimizer, build_scheduler, build_criterion
    optimizer = build_optimizer(model, **parser.optimizer)
    scheduler = build_scheduler(optimizer, **parser.scheduler)
    criterion = build_criterion(**parser.criterion)

    from engine import build_engine
    engine_ = build_engine(model, optimizer, scheduler, criterion, logger,
                           **parser.engine)
    engine_.train(loader_train, loader_test, **parser.train)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "not matched argument enter your cfg path and log path(default path: ./log)"
        )

    sys.path.append(os.path.dirname(__file__))
    from common import Parser, Logger

    parser = Parser(sys.argv[1], __debug__)
    logger = Logger(sys.argv[2]) if len(sys.argv) == 3 else Logger()

    main(parser, logger)
