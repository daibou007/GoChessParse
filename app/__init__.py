import os
from flask import Flask
from flask_bootstrap import Bootstrap


def create_app():
    '''
    创建 Flask 应用实例
    :return: Flask app
    '''
    app = Flask(__name__)
    bootstrap = Bootstrap()
    bootstrap.init_app(app)
    
    # 配置
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'Hello,Nicapoet'
    app.config['PORT'] = int(os.environ.get('PORT', 5001))  # 设置默认端口为5001
    
    # 注册蓝图
    from .web_api import upload_blueprint
    app.register_blueprint(upload_blueprint)
    
    return app
