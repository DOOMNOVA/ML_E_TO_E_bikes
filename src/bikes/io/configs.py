#parse, merge and convert config objects

import typing as T
import omegaconf as oc

#types
Config = oc.ListConfig | oc.DictConfig

def parse_file(path:str) -> Config:
    """Parse a config file from path
    
    Args: path(str) : path to local config
    
    Returns: Config: representation of config file
    
    """
    
    return oc.OmegaConf.load(path)


def parse_string(string:str)-> Config:
    """Parse the given config string
    
    Args: string(str): content of config string
    
    Returns: 
    Config: representation of the config string.
    
    """
    return oc.OmegaConf.create(string)

#Mergers

def merge_configs(configs: T.Sequence[Config]) -> Config:
    """"Merge a list of config into a single config
    
    Args:
    configs (T.Sequence[Config]): list of configs.
    Returns :
    Config: representation of merged config objects
    
    """
    return oc.OmegaConf.merge(*configs)
    
    #converters
    
def to_object(config:Config, resolve:bool=True) -> object:
    """Convert a config object to a python object.
    
    Args:
        config(Config): representation of the config.
        resolve(bool): resolve variables. Defaults to True
        
    
    Returns:
        object: conversion of the config to a python object
    
    
    """
    return oc.OmegaConf.to_container(config,resolve=resolve)

    
    