import flask
from flask import request
from web_service.BaseService import BaseService
import web_service
from pathlib import Path
import logging
import torch
from config.configuration import parse_config, ServiceConfig
logger = logging.getLogger("audio_service")
logger.setLevel(logging.DEBUG)

app = flask.Flask(__name__)
app.config["DEBUG"] = True

SERVICE_CONFIG_FILEPATH = "./build/service.cfg"
MODEL_CHECKPOINT_FILEPATH = "./build/model.ckpt"
MODEL_HPARAMS_FILEPATH = "./build/hparams.yaml"
MODEL_CONFIG_FILEPATH = "./build/config.cfg"
CLASS_LIST_FILEPATH = "./build/class-list.csv"
WORKING_DIRECTORY = "./working_directory"



def importService(name) -> BaseService:
    components = name.split(".")
    mod = web_service
    for comp in components:
        mod = getattr(mod, comp)
    return mod


@app.route("/", methods=["GET"])
def home():
    if request.args.get("file") is None:
        error = {
            "error": {
                "code": "received_no_files",
                "message": "Did not receive any files",
            }
        }
        return flask.jsonify(error), 400
    try:
        pre_proccessed_data = service.pre_proccess_data(request.args)
        predictions = service.inference(pre_proccessed_data)
        response = service.post_process(predictions, pre_proccessed_data, request.args)
    except Exception as e:
        torch.cuda.empty_cache()
        error = {"error": {"code": 500, "message": str(e),}}
        raise e
        return flask.jsonify(error), 500
    torch.cuda.empty_cache()
    return flask.jsonify(response)


@app.route("/class-list", methods=["GET"])
def classes():
    return flask.jsonify(service.class_list["latin_name"].tolist())

#load service configuration
service_config:ServiceConfig = parse_config(Path(SERVICE_CONFIG_FILEPATH), enviroment_prefix = "SERVICE_", config_type=ServiceConfig)

# if __name__ == "__main__":
ServiceClass = importService(service_config.service_class_name)
service = ServiceClass(
    model_class_name=service_config.model_class_name,
    model_config_filepath=MODEL_CONFIG_FILEPATH,
    model_checkpoint_filepath=MODEL_CHECKPOINT_FILEPATH,
    model_hparams_filepath=MODEL_HPARAMS_FILEPATH,
    class_list_filepath=CLASS_LIST_FILEPATH,
    working_directory=WORKING_DIRECTORY,
    service_config = service_config
)
# app.run(port=5000)
# print("end")
