from modules.agents import classifier_agent
from importlib import import_module
import argparse


def get_parsed_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Training. Currently only supports audio classification')
    parser.add_argument('-c', '--config_file',
                        default="configs/urban_sound_8k_config.py",
                        help='Config file for agent.\n' +
                        'Default: configs/urban_sound_8k_config.py')
    args = parser.parse_args()
    # remove / and .py from config path
    args.config_file = args.config_file.replace('/', '.')[:-3]
    return args


def main():
    args = get_parsed_args()
    config_module = import_module(args.config_file)
    agent = classifier_agent.ClassifierAgent(config_module.CONFIG)

    agent.train()
    agent.finalize_exit()


if __name__ == "__main__":
    main()
