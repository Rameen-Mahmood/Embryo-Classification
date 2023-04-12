import argparse 

def args_parser():
    parser = argparse.ArgumentParser(description="arguments")

    parser.add_argument("--p", 
    	dest="path", action="store", 
    	default = "/media/nyuad/189370B3586E6F7C/group1", 
    	help="indicate the path of datasets")
    parser.add_argument("--m", dest="model", type=str, action="store", 
    	default = 'resnet50', 
    	help="choose the learning model")
    parser.add_argument('--mode', type=str, default="train",
        help='mode: train or test')
    parser.add_argument('--bs', dest='batch_size', type=int, default=32,
        help='batch size')
    parser.add_argument('--ep', dest='num_epochs', type=int, default=50,
        help='number of epochs')
    parser.add_argument('--save_dir', type=str, 
    	help='Directory relative which all output files are stored',
        default='checkpoints')

    return parser
