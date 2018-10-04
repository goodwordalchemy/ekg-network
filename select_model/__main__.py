'''
Assumptions;
* config must contain a "ModelName" parameter, which names a module in the "models" directory
* the model module must contain a function called "get_model"
* the config file must contain a "ParameterSpecification" field, that tells the parameter search script what parameters to try."
* the 'SearchMethod' field should name one of the modules in the search methods directory
'''
from importlib import import_module

from config import get_config

MODELS_MODULE_PATH = 'models.'
SEARCH_METHODS_MODULE_PATH = 'select_model.search_methods.'

def _import_from(module, name):
    module = import_module(module)
    method = getattr(module, name)

    return method



def main():
    '''
    get config file from positional argument'

    extract the model from the config file

    extract the parameters from the config file

    run the correct search method script with the correct model and parameters
    '''

    config = get_config()

    model_name = config['ModelName']
    model_name = MODELS_MODULE_PATH + model_name
    create_model_function = _import_from(model_name, 'create_model')

    parameter_specification = config['ParameterSpecification']

    search_method_name = config['SearchMethod']
    search_method_name = SEARCH_METHODS_MODULE_PATH + search_method_name
    search_method_function = _import_from(search_method_name, 'main')

    search_method_function(create_model_function, parameter_specification)


if __name__ == '__main__':
    main()
