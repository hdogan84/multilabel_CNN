from pathlib import Path
import json

from numpy import single


class ResultLogger:
    result_dict = {}
    segments = []
    class_list = []
    version = None
    model_name = None
    output_path = ""

    def __init__(
        self, segments, model_name="unkown", class_list=[], version=None, output_path=""
    ):
        self.class_list = class_list
        self.version = version
        self.segments = segments
        self.model_name = model_name
        self.output_path = output_path

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
            filepath = segment["filepath"].as_posix()
            start_time = segment["start_time"]
            end_time = segment["end_time"]

            if filepath not in self.result_dict:
                self.result_dict[filepath] = {
                    "modelName": self.model_name,
                    "version": self.version,
                    "fileId": segment["filepath"].stem,
                    "classIds": self.class_list,
                    "channels": [[]] * len(segment_result["predictions"]),
                }
            file_entry = self.result_dict[filepath]
            for channel, prediction in enumerate(segment_result["predictions"]):
                file_entry["channels"][channel].append(
                    {
                        "startTime": start_time,
                        "endTime": end_time,
                        "predictions": {"probabilities": prediction,},
                        "groudTruth": segment_result["groundTruth"][channel],
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
