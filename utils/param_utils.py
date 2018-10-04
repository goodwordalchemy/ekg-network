def params_dict_to_str(params_dict):
    pstring = ''

    for key, param in sorted(params_dict.items()):
        pstring += '{}_{}__'.format(key, str(param))

    pstring = pstring[:-2]

    return pstring
