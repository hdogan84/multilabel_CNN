import torch
import sys
from torchmetrics import Accuracy, AveragePrecision, F1, AUC

from tools.config import load_yaml_config
from tools.tensor_helpers import pool_by_segments
from tools.result_logger import ResultLogger
from tools.tensor_helpers import (
    transform_class_tensor,

)


def progbar(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')
    if(curr == total):
        print('\n')
    sys.stdout.flush()
    

class RunBaseTorchScriptModel:
    metrics = False
    result_file = None
    config = None
    result_file = False
    result_filepath = None
    result_logger = None

    # set by setup method
    transform_signal = None
    transform_image = None
    data_loader = None

    # set by
    def __init__(
        self,
        model_filepath,
        config_filepath,
        validation=False,
        result_file=True,
        result_filepath="predictions.csv",
        device="cpu",
        binary_threshold=0.5
    ):
        self.config = load_yaml_config(config_filepath)
        self.model_filepath = model_filepath
        self.validation = validation
        self.result_file = result_file
        self.result_filepath = result_filepath
        self.result_logger = ResultLogger() if result_file else None
        self.device = device
        self.binary_threshold = binary_threshold

    def setup_transformations(self): 
        # has to return transformation_pipline signal, transformation_pipeline image
        return None, None
    def setup_dataloader(self, transform_signal, transform_image):
        # has to return dataloader, data_list [{filepath, start,channel}], num_classes, class_list
        # data_loader,data_list, num_classes, class_list
        return None, None, 0, None
        

    def setup_class_tensor_transform_matrix(self, class_list):
        # has to return class tensor transformation matrix
        return None
    
    def setup_validation(self,num_classes): 
        # has to return a dictionary of metrics function, Key will be used as key for the results
        metrics_dict = {
                    "accuracy" : Accuracy(num_classes=num_classes,threshold=self.binary_threshold).to(self.device) if self.validation else None,
                    "f1" : F1(num_classes=num_classes,threshold=self.binary_threshold).to(self.device) if self.validation else None,
                    "averagePrecision" : AveragePrecision(num_classes=num_classes).to(self.device) if self.validation else None  
                }
        return metrics_dict
    

    def validation_step(self,metrics_dict,  predictions, classes):
        # every batch this step is called
        target = classes.type(torch.int)

        result = {}
        for key in metrics_dict:
            result[key] = metrics_dict[key](predictions, target)
        return result
    def validation_end(self, metrics_dict):
        result = {}
        for key in metrics_dict:
            result[key] = metrics_dict[key].compute().item()
        return result
       
    def run(self
    ):
        transform_signal,transform_image = self.setup_transformations()
        data_loader,data_list, num_classes, class_list = self.setup_dataloader(transform_signal,transform_image)
        class_tensor_transformation_matrix = self.setup_class_tensor_transform_matrix(class_list)
     
        # class_tensor_transformation_matrix.to(self.device)
        metrics_dict = self.setup_validation(num_classes) if self.validation else None

        with torch.no_grad():
            model = torch.jit.load(self.model_filepath, map_location=self.device)
            batch_results = []
            batch_amount = len(data_loader)

            for index, batch in enumerate(data_loader):

                x, classes, segment_indices = batch
                classes = classes.to(self.device)
                x = x.to(self.device)
                segment_indices = segment_indices

                preds = model(x)

                if( class_tensor_transformation_matrix is not None):
                    preds = transform_class_tensor(
                        preds, class_tensor_transformation_matrix
                    ) 
                predictions = preds
                # pool segments
                # pool classes only if validation
                if(self.validation):
                    classes, _ = pool_by_segments(classes, segment_indices)
                    
                predictions, _ = pool_by_segments(
                    predictions, segment_indices, pooling_method="max"
                )
                if self.result_file:
                    # tmp_segment_indices = pool_by_segments(segment_indices, segment_indices)
                    self.result_logger.add_batch(predictions, segment_indices, data_list)

                if self.validation:
                    batch_result = self.validation_step(metrics_dict, predictions, classes)
                    batch_results.append(batch_result)
                progbar(index + 1, batch_amount,20)

            if self.validation:
                validation_result = self.validation_end(metrics_dict)
                print(validation_result)
                
            if self.result_file:
                self.result_logger.write_results_to_file(self.result_filepath)

