import yaml
import os

def get_bfp_args():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    with open(os.path.join(__location__, 'bfp_config.yaml')) as file:
        try:
            hbfpConfig = yaml.safe_load(file)
            print(hbfpConfig['hbfp'])   
            return hbfpConfig['hbfp']
        except yaml.YAMLError as exc:
            print(exc)

if __name__ == '__main__':
    get_bfp_args()