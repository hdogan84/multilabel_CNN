#!/usr/bin/python
# -*- coding: utf-8 -*-

# sources: 
    # https://blog.miguelgrinberg.com/post/designing-a-restful-api-with-python-and-flask
    # http://flask.pocoo.org/docs/1.0/patterns/fileuploads/

'''
# Info/Help

# Run docker
docker run -d -p 4000:80 --ipc=host --name birdid18-8-flask-v03 --rm birdid18-8-flask-v03

# Mounts /net/mfnstore-lin/export/tsa_transfer/BirdId/_UserContentWeb into /workspace/data
docker run -d -p 4000:80 -v /net/mfnstore-lin/export/tsa_transfer/BirdId/_UserContentWeb:/workspace/data --ipc=host --name birdid18-8-flask-v03 --rm birdid18-8-flask-v03

# Run docker without starting flaskTest.py (to edit/try new versions of flaskTest.py)
docker run -it -p 4000:80 -v /net/mfnstore-lin/export/tsa_transfer/BirdId/_UserContentWeb:/workspace/data --ipc=host --name birdid18-8-flask-v03 --rm birdid18-8-flask-v03 /bin/bash

#####  HOW TO  #####

# Identify in browser via file upload
http://localhost:4000/

# Identify by posting file
curl http://localhost:4000/identify -X POST -F "file=@/path/to/file.mp3"

# Identify by passing link (subdir and filename) to file in mounted volume
curl "http://localhost:4000/identify?subdir=00_Test&filename=AcrSci00002.mp3"

# or in browser: http://localhost:4000/identify?subdir=00_Test&filename=AcrSci00002.mp3

# Add arg save_predictions=0 to not save segment or file based predictions as .npy files
curl "http://localhost:4000/identify?subdir=00_Test&filename=AcrSci00002.mp3&save_predictions=0"

# Add arg naturblick_style=1 to return output in Natublick 2019 style
curl "http://localhost:4000/identify?subdir=00_Test&filename=AcrSci00002.mp3&naturblick_style=1"

# Get some info / metadata
http://localhost:4000/classIds
http://localhost:4000/metadata
http://localhost:4000/birds/german
http://localhost:4000/birds/english
http://localhost:4000/birds/scientific
http://localhost:4000/birds/number

'''

from flask import Flask, jsonify, flash, request, redirect, url_for, send_from_directory, make_response
from werkzeug.utils import secure_filename
import json
import os


from predictSingleFile import *


########  Config  ########

debug_mode = False #False #True
port = 80 # Use 80 when buidlding docker image !


# To temporary upload files passed via curl
#UploadDirTemp = '_UploadDirTemp/'
UploadDirTemp = '_UploadDirTemp/'
if not os.path.exists(UploadDirTemp): os.makedirs(UploadDirTemp)

# Mount point (e.g. via -v /net/mfnstore-lin/export/tsa_transfer/BirdId/_UserContentWeb:/workspace/data)
UploadRootDir = 'data/'


##########################


app = Flask(__name__)


if debug_mode:
    app.config['JSON_AS_ASCII'] = False # "Drosselrohrs\u00e4nger" --> "Drosselrohrsänger" (possible security risc!)


def toJson(obj):
    if debug_mode:
        return jsonify(obj)
    else:
        return json.dumps(obj, ensure_ascii=False) # no need for app.config['JSON_AS_ASCII'] = False

def uploadPostedFile(request):

    # Check if post request has file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        AudioFid = os.path.join(UploadDirTemp, filename)
        file.save(AudioFid)

        return AudioFid

def cleanUploadDirTemp(AudioFid):
    FileName = os.path.basename(AudioFid)
    for f in os.listdir(UploadDirTemp):
        if f.startswith(FileName[:-4]):
            os.remove(os.path.join(UploadDirTemp, f))


@app.route('/identify', methods=['GET', 'POST'])
def processRequest():

    # Identify by posting file
    # curl http://localhost:4000/identify -X POST -F "file=@/path/to/file.mp3"
    if request.method == 'POST':
        AudioFid = uploadPostedFile(request)
        Predictions = identify(AudioFid)
        Output = postProcessPredictions(Predictions, AudioFid)
        cleanUploadDirTemp(AudioFid) # Remove files in UploadDirTemp
        return toJson(Output)

    # Identify by passing link (subdir and filename) to file in mounted volume
    # curl "http://localhost:4000/identify?subdir=00_Test&filename=AcrSci00002.mp3"
    # or in browser: http://localhost:4000/identify?subdir=00_Test&filename=AcrSci00002.mp3
    subdir = request.args.get('subdir', default='./', type=str)
    filename = request.args.get('filename', default='', type=str)

    # Add arg save_predictions=0 to not save segment or file based predictions as .npy files
    # curl "http://localhost:4000/identify?subdir=00_Test&filename=AcrSci00002.mp3&save_predictions=0"
    save_predictions = request.args.get('save_predictions', default='1', type=str)
    if save_predictions == '1' or save_predictions == 'True' or save_predictions == 'true':
        SavePredictions = True
    else:
        SavePredictions = False

    # # Add arg naturblick_style=1 to return output in Natublick 2019 style
    # # curl "http://localhost:4000/identify?subdir=00_Test&filename=AcrSci00002.mp3&naturblick_style=1"
    # naturblick_style = request.args.get('naturblick_style', default='0', type=str)
    # if naturblick_style == '1' or naturblick_style == 'True' or naturblick_style == 'true':
    #     OutputInNaturblickStyle2019 = True
    # else:
    #     OutputInNaturblickStyle2019 = False

    # Add arg to format output
    # curl "http://localhost:4000/identify?subdir=00_Test&filename=AcrSci00002.mp3&out_style=naturblick2019"
    OutputStyle = 'default'
    OutputStyle = request.args.get('out_style', default='default', type=str)


    if filename == '':
        return redirect(url_for('upload_file_browser'))
    

    filename = secure_filename(filename)
    AudioFid = os.path.join(UploadRootDir + subdir, filename)
    
    #Output = identify(AudioFid, SavePredictions=SavePredictions, OutputInNaturblickStyle2019=OutputInNaturblickStyle2019)

    Predictions = identify(AudioFid)
    writeCsvFile(Predictions, AudioFid)
    #Output = postProcessPredictions(Predictions, AudioFid, SavePredictions=SavePredictions, OutputInNaturblickStyle2019=OutputInNaturblickStyle2019)
    Output = postProcessPredictions(Predictions, AudioFid, SavePredictions=SavePredictions, OutputStyle=OutputStyle)
    return toJson(Output)


# Run in browser to upload audio file via form: http://localhost:4000/
@app.route('/', methods=['GET', 'POST'])
def upload_file_browser():

    if request.method == 'POST':

        AudioFid = uploadPostedFile(request)
        Predictions = identify(AudioFid)
        Output = postProcessPredictions(Predictions, AudioFid, SavePredictions=False, OutputStyle='default')

        cleanUploadDirTemp(AudioFid) # Remove files in UploadDirTemp

        writeCsvFile(Predictions, AudioFid)


        HtmlStr = '<!doctype html>'
        HtmlStr += '<title>Bird Identification</title>'
        HtmlStr += '<h1>Upload Audio File</h1>'
        HtmlStr += '<form method=post enctype=multipart/form-data>'
        HtmlStr += '<input type=file name=file>'
        HtmlStr += '<input type=submit value=Upload>'
        HtmlStr += '</form>'

        HtmlStr += '<h3>Results</h3>'
        HtmlStr += '<ol>'
        HtmlStr += '<li><a target="_blank" rel="noopener noreferrer" href="' + RefSysLinkPrefix + Output[0]['scieName'].replace(' ', '%20') + '">' + Output[0]['gerName'] + '</a>&nbsp;(' + Output[0]['scieName'] + ')&nbsp;&nbsp;[' + Output[0]['prediction'] + '%]</li>'
        HtmlStr += '<li><a target="_blank" rel="noopener noreferrer" href="' + RefSysLinkPrefix + Output[1]['scieName'].replace(' ', '%20') + '">' + Output[1]['gerName'] + '</a>&nbsp;(' + Output[1]['scieName'] + ')&nbsp;&nbsp;[' + Output[1]['prediction'] + '%]</li>'
        HtmlStr += '<li><a target="_blank" rel="noopener noreferrer" href="' + RefSysLinkPrefix + Output[2]['scieName'].replace(' ', '%20') + '">' + Output[2]['gerName'] + '</a>&nbsp;(' + Output[2]['scieName'] + ')&nbsp;&nbsp;[' + Output[2]['prediction'] + '%]</li>'
        HtmlStr += '</ol>'

        # NNN
        # Provide link to results csv file
        FileName = os.path.basename(AudioFid)[:-4] + '.csv'

        HtmlStr += '<br>'
        HtmlStr += '<a href="' + url_for('download', filename=FileName) + '">' + FileName + '</a>'

        return HtmlStr
        
    return '''
    <!doctype html>
    <title>Bird Identification</title>
    <h1>Upload Audio File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):

    # Offer file in UploadDirTemp for download
    return send_from_directory(directory=UploadDirTemp, filename=filename, as_attachment=True)

    # # Read csv from file, write csv object, delete csv file
    # si = io.StringIO()
    # cw = csv.writer(si, delimiter=';')

    # with open(UploadDirTemp + filename, newline='') as f:
    #     cr = csv.reader(f)
    #     for row in cr:
    #         l = row[0].split(CsvDelimiter) # ['ParMaj0;FriCoe0;LusMeg0'] --> ['ParMaj0';'FriCoe0';'LusMeg0']
    #         cw.writerow(l)

    # # Remove csv file
    # os.remove(os.path.join(UploadDirTemp, filename))

    # output = make_response(si.getvalue())
    # output.headers["Content-Disposition"] = "attachment; filename=" + filename
    # output.headers["Content-type"] = "text/csv"
    # return output



@app.route("/classIds")
@app.route("/ClassIds")
def getClassIds():
    return toJson(ClassIds.tolist())


@app.route("/birds")
@app.route("/metadata")
def getBirds():
    return toJson(BirdsEurope254MetadataDict)


@app.route("/birds/german")
def getBirdsGerman():
    birds = []
    for class_id in ClassIds:
        birds.append(BirdsEurope254MetadataDict[class_id]['NameDt'])
    return toJson(birds)

@app.route("/birds/english")
def getBirdsEnglish():
    birds = []
    for class_id in ClassIds:
        birds.append(BirdsEurope254MetadataDict[class_id]['NameEn'])
    return toJson(birds)

@app.route("/birds/scientific")
def getBirdsLat():
    birds = []
    for class_id in ClassIds:
        birds.append(BirdsEurope254MetadataDict[class_id]['NameLat'])
    return toJson(birds)


@app.route("/birds/number")
def getBirdsNumber():
    return toJson(len(BirdsEurope254MetadataDict))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=debug_mode)

