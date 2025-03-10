import os
import json
import time
from flask import render_template,jsonify
from werkzeug.utils import secure_filename
from .forms import ImageForm
from . import upload_blueprint
from ..cv_parse import output_result

UPLOAD_FOLDER = os.path.abspath(os.path.dirname(__file__)) + "/upload/"


@upload_blueprint.route('/', methods=['GET', 'POST'])
def upload():
    form = ImageForm()
    if form.validate_on_submit():
        try:
            image_file = form.image.data
            # 修正文件名获取方式
            original_filename = image_file.filename
            filename = secure_filename(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '_' + original_filename)
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            
            # 保存文件
            image_file.save(image_path)
            
            output = output_result(image_path, False)
        except ZeroDivisionError:
            return jsonify({'ret': -1, 'msg': 'detect fail'})
        except Exception as e:
            return jsonify({'ret': -1, 'msg': f'upload failed: {str(e)}'})
        else:
            print('POST output', output)
            return jsonify({'ret': 1, 'msg': 'detect succeed', 'output': output.tolist()})
    return render_template('upload.html', form=form)
