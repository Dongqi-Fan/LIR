from copy import deepcopy
from utils import get_root_logger


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    if opt['model_type'] == 'ConStyle':
        from models.ConStyle_model import ConStyleModel as Model

    elif opt['model_type'] == 'Origin':
        from models.origin_model import OriginModel as Model

    else:
        raise NotImplementedError('Model type [{:s}] is not defined.'.format(opt['model_type']))

    model = Model(opt)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model


