from espnet2.tasks.kws import KwsTask

def get_parser():
    parser = KwsTask.get_parser()
    return parser

def main(cmd=None):
    r"""KWS training.

    Example:

        % python kws_train.py kws --print_config --optim adadelta \
                > conf/train_kws.yaml
        % python kws_train.py --config conf/train_kws.yaml
    """
    KwsTask.main(cmd=cmd)


if __name__ == "__main__":
    main() 