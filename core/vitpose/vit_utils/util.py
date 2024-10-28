from importlib import import_module
import warnings


MODEL_ABBR_MAP = {
    's': 'small',
    'b': 'base',
    'l': 'large',
    'h': 'huge'
}


def dyn_model_import(dataset: str, model: str):
    config_name = f'core.vitpose.cfg.ViTPose_{dataset}'
    imp = import_module(config_name)
    model = f'model_{MODEL_ABBR_MAP[model]}'
    return getattr(imp, model)


def resize(inputs,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = int(inputs.shape[0]), int(inputs.shape[1])
            output_h, output_w = int(size[0]), int(size[1])
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')