from web_service.BaseService import BaseService


class AudioService(BaseService):
    def pre_proccess_data(self, params):
        print("AudioSerice pre_proccess_data")

    def post_process(self, data):
        print("AudioSerice post_process")
