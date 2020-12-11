from __future__ import absolute_import, division
import configparser
from os import environ
import os
from os.path import abspath, dirname, join


def to_environ_key(key):
    return key.upper()


class PrivateConfigurator(object):
    """
    Singleton class for private configuration data
    """
    __instance = None

    def __new__(cls):
        if PrivateConfigurator.__instance is None:
            PrivateConfigurator.__instance = object.__new__(EnvironmentConfigParser)
            PrivateConfigurator.__instance.__init__(interpolation=configparser.ExtendedInterpolation())
            PrivateConfigurator.__instance.read(get_private_config_path())
        return PrivateConfigurator.__instance


class Configurator(object):
    """
    Singleton class for user configuration data
    """
    __instance = None
    __cfg_path = None

    def __new__(cls, cfg_path):
        if Configurator.__instance is None:
            Configurator.__instance = object.__new__(EnvironmentConfigParser)
            Configurator.__instance.__init__(interpolation=configparser.ExtendedInterpolation())
            Configurator.__instance.read(cfg_path)
        return Configurator.__instance


class EnvironmentConfigParser(configparser.ConfigParser):
    """
    ConfigParser with additional option to read from environment variables
    """
    def has_option(self, section, option):
        if to_environ_key('_'.join((section, option))) in environ:
            return True
        return super(EnvironmentConfigParser, self).has_option(section, option)

    def get(self, section, option, raw=False, **kwargs):
        key = to_environ_key('_'.join((section, option)))
        if key in environ:
            return environ[key]
        return super(EnvironmentConfigParser, self).get(section, option, raw=raw, **kwargs)


def readConfigFile(cfgfile):
    """
    Read config files and return ConfigParser object

    @param cfgfile: filename or array of filenames
    @return: ConfigParser object
    """
    parser = EnvironmentConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    parser.read(cfgfile)
    return parser


def get_private_config_path():
    """
    Get absolute path to the private config file
    """
    current_path = abspath(dirname(__file__))
    conf_path = join(current_path, 'etc/private_config.cfg')

    if conf_path:
        return conf_path

    print("Private config file not found")


def get_config_path():
    """
    Get absolute path to the config file
    """
    current_path = abspath(dirname(__file__))
    conf_path = join(current_path, 'etc/Public/config.cfg')

    if conf_path:
        return conf_path

    print("Config file not found")
