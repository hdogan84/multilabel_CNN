from pathlib import Path
import json
import math
import pickle

from numpy import single

import torch


def prob_to_logit(p):
    return torch.logit(torch.tensor(p)).item()


def logit_to_prob(logit):
    return torch.sigmoid(torch.tensor(logit)).item()


class ResultLogger:
    result_dict = {}
    segments = []
    class_list = []
    version = None
    model_name = None
    output_path = ""
    output_type = "probabilities"

    def __init__(
        self,
        segments,
        model_name="unkown",
        class_list=[],
        version=None,
        output_path="",
        output_type="probabilities",
    ):
        self.class_list = class_list
        self.version = version
        self.segments = segments
        self.model_name = model_name
        self.output_path = output_path
        if output_type != "probabilities" and output_type != "logits":
            raise Exception("Unkown output type {}".format(output_type))
        self.output_type = output_type

    def validation_end(self, metrics_dict):
        result = {}
        for key in metrics_dict:
            values = metrics_dict[key].compute()
            if isinstance(values, list):
                values = [x.item() for x in values]
            else:
                values = values.item()
            result[key] = values
        out_filepath = Path(self.output_path).joinpath(
            Path(
                "./{}-{}/{}.json".format(self.model_name, self.version, "metrics.json")
            )
        )

        out_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(out_filepath.as_posix(), "w") as outfile:
            json.dump(result, outfile)

    def log_batch(self, prediction, ground_truth, segment_indices, batch_index):
        prediction_list = prediction.tolist()
        ground_truth_list = ground_truth.tolist()
        segment_indices_list = segment_indices.tolist()
        # group predictions by segment indeces to find all channels all channels are(and have to be) always in one batch
        grouped_values = {}
        for num, index in enumerate(segment_indices_list):
            if index not in grouped_values:
                grouped_values[index] = {
                    "predictions": [],
                    "segment": self.segments[index],
                    "groundTruth": [],
                }
            grouped_values[index]["predictions"].append(prediction_list[num])
            grouped_values[index]["groundTruth"].append(ground_truth_list[num])
        for _, segment_result in grouped_values.items():
            segment = segment_result["segment"]
            predictions = segment_result["predictions"]
            filepath = segment["filepath"].as_posix()
            start_time = segment["start_time"]
            end_time = segment["end_time"]

            if filepath not in self.result_dict:
                self.result_dict[filepath] = {
                    "modelName": self.model_name,
                    "version": self.version,
                    "fileId": segment["filepath"].stem,
                    "classIds": self.class_list,
                    "channels": [[] for _ in range(len(predictions))],
                }
            file_entry = self.result_dict[filepath]
            for channel, prediction in enumerate(predictions):
                probabilities = []
                logits = []
                # print(predictions)
                if self.output_type == "probabilities":
                    probabilities = prediction
                    logits = [prob_to_logit(x) for x in prediction]
                if self.output_type == "logits":
                    logits = logits
                    probabilities = [logit_to_prob(x) for x in prediction]
                file_entry["channels"][channel].append(
                    {
                        "startTime": start_time,
                        "endTime": end_time,
                        "predictions": {
                            "probabilities": probabilities,
                            "logits": logits,
                        },
                        "groundTruth": segment_result["groundTruth"][channel],
                    }
                )

    def write_to_json(self, single_files=True):
        if single_files:
            for _, file_dict in self.result_dict.items():
                out_filepath = Path(self.output_path).joinpath(
                    Path(
                        "./{}-{}/{}.json".format(
                            self.model_name, self.version, file_dict["fileId"]
                        )
                    )
                )
                out_filepath.parent.mkdir(parents=True, exist_ok=True)
                with open(out_filepath.as_posix(), "w") as outfile:
                    json.dump(file_dict, outfile)
        else:
            with open("results.json", "w") as outfile:
                json.dump(self.result_dict, outfile)

    def saveAsPickle(self, single_files=True):

        for _, file_dict in self.result_dict.items():
            result_dict = {}
            result_dict["modelName"] = self.model_name
            result_dict["version"] = self.version
            result_dict["file_id"] = file_dict["fileId"]
            out_filepath = Path(self.output_path).joinpath(
                Path(
                    "./{}-{}/{}.json".format(
                        self.model_name, self.version, file_dict["fileId"]
                    )
                )
            )
            out_filepath.parent.mkdir(parents=True, exist_ok=True)
            (n_channels, n_segments, n_classes) = predictions.shape
            result_dict["n_channels"] = n_channels
            result_dict["n_segments"] = n_segments
            result_dict["n_classes"] = n_classes

            result_dict["segment_duration"] = cfg.SIG_LENGTH

            result_dict["class_ids_birdnet"] = cfg.LABELS

            # latin names (first part of birdnet labels)
            result_dict["class_ids"] = []
            for ix in range(len(cfg.LABELS)):
                class_str = cfg.LABELS[ix]
                class_arr = class_str.split("_")
                class_la = class_arr[0]
                class_en = class_arr[1]
                result_dict["class_ids"].append(class_la)

            result_dict["start_times"] = []
            start_time = 0.0
            for segm_ix in range(n_segments):
                result_dict["start_times"].append(start_time)
                # print(start_time)
                start_time += cfg.SIG_LENGTH - cfg.SIG_OVERLAP

            # print(predictions.dtype) # float32
            result_dict["prediction_logits"] = predictions.astype(np.float32)
            result_dict["prediction_probabilities"] = model.flat_sigmoid(
                predictions
            ).astype(np.float32)

            # Save as pickle
            with open(path, "wb") as handle:
                pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

