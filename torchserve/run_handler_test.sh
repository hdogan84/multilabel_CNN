#/bin/bash
cd .. && torch-model-archiver --serialized-file ./build/model.pt --export-path ./build --handler ./build/handler.py --force --model-name test --version 1 --extra-files ./build/index_to_name.json,./build/config.yaml,./build/base_audio_handler.py && \
cd torchserve && \
# docker-compose restart && \
curl -X DELETE http://localhost:8081/models/test/1
sleep 0.5 && \
curl -X POST "http://localhost:8081/models?url=test.mar&initial_workers=1" && \
sleep 0.5 && \
curl http://localhost:8080/predictions/test/1 -F "data=@./test-assets/test.wav"