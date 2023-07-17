def get_model(cfg, logger):
    if cfg.arch == 'InvEDRS_arb':
        from models.inv_arb_edrs import InvArbEDRS as Model
        model = Model(cfg)     
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model
