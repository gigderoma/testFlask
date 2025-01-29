
"""BluePrint/Flask Init file for Application"""

#Pylint
# pylint: disable=import-outside-toplevel
# pylint: disable=import-error
# pylint: disable=ungrouped-imports
# pylint: disable=wrong-import-order

import sys
from flask import Flask, request, current_app


from flask_bootstrap import Bootstrap
from flask_moment import Moment

from config import Config
from flask.logging import logging,default_handler



bootstrap = Bootstrap()


def create_app(config_class=Config):
    """Method that intializes whole app for flask"""
    app = Flask(__name__)
    # app.config['DEBUG'] = True
    app.config.from_object(config_class)
    #Check if we are running in Kubernetes by getting environment variables
    if app.config['KUBERNETES_SERVICE_PORT'] != "" :
        #If Environment variable is not set we are not running in kubernetes
        #We will change the default handler to log to stdout
        app.logger.removeHandler(default_handler)
        app.logger.addHandler(logging.StreamHandler(sys.stdout))        
    #We also want to check if we are running with Gunicorn and use logging configuration from it.
    if app.config['GUNICORN_LOGGER'].handlers != []:
        app.logger.handlers = app.config['GUNICORN_LOGGER'].handlers
        app.logger.setLevel(app.config['GUNICORN_LOGGER'].level)
    bootstrap.init_app(app)
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)
    return app
