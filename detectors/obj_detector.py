import yaml

def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)



class HumanDetector(object):
    def __init__(self, weight="", data=None):
        pass

        # model load with weight

        # classes names
        names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}

        # warm up

    def preprocess(self):
        pass
    def detect(self):
        pass

    def postprocess(self):
        pass