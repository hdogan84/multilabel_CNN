import flask
from flask import request
from web_service.BaseService import BaseService
import web_service
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)

app = flask.Flask(__name__)
app.config["DEBUG"] = True

SERVICE_CLASS_NAME = "AudioService"
MODEL_CLASS_NAME = "CnnBirdDetector"
MODEL_CHECKPOINT_FILEPATH = "../data/val_f1_score=0.85-epoch=23.ckpt"
MODEL_HPARAMS_FILEPATH = "../data/hparams.yaml"
MODEL_CONFIG_FILEPATH = "../data/config.cfg"
CLASS_LIST_FILEPATH = "../class-list.csv"
WORKING_DIRECTORY = "./working_directory"


def importService(name) -> BaseService:
    components = name.split(".")
    mod = web_service
    for comp in components:
        mod = getattr(mod, comp)
    return mod


@app.route("/", methods=["GET"])
def home():
    data = service.pre_proccess_data(request.args)
    prediction = service.inference(data)
    response = service.post_process(prediction)
    return response


# if __name__ == "__main__":
ServiceClass = importService(SERVICE_CLASS_NAME)
service = ServiceClass(
    model_class_name=MODEL_CLASS_NAME,
    model_checkpoint_filepath=MODEL_CHECKPOINT_FILEPATH,
    model_hparams_filepath=MODEL_HPARAMS_FILEPATH,
    class_list_filepath=CLASS_LIST_FILEPATH,
    working_directory=WORKING_DIRECTORY,
)
app.run(port=5666)
