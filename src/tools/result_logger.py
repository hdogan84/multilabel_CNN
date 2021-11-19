import csv
import os
import sys

from torch.utils import data


class ResultLogger:
    result_dic = {}
    headline = ["filepath", "filename", "channel", "start_time", "end_time", "results"]

    def add_result(self, filepath, channel, start_time, end_time, results):

        if not filepath in self.result_dic:
            self.result_dic[filepath] = []
        self.result_dic[filepath].append(
            [
                filepath,
                os.path.basename(filepath),
                channel,
                start_time,
                end_time,
                results,
            ]
        )

    def add_batch(self, predictions, segment_indices, data_list):
        for preds, sg_ix in zip(predictions.tolist(), segment_indices.tolist()):
            start_time = data_list[sg_ix]["start_time"]
            end_time = data_list[sg_ix]["end_time"]
            filepath = data_list[sg_ix]["filepath"]
            channel = data_list[sg_ix]["channel"]
            self.add_result(filepath.as_posix(), channel, start_time, end_time, preds)

    def get_file_results(self, filepath):
        return self.result_dic[filepath]

    def write_results_to_file(self, filepath):
        with open(filepath, mode="w") as filewriter:
            writer = csv.writer(
                filewriter, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow(self.headline)
            for key in self.result_dic:
                writer.writerows(self.result_dic[key])

