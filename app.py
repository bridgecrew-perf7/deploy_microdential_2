import flask
from flask.wrappers import Request
from werkzeug.wrappers import request
from data.builder import *
from flask_wtf import FlaskForm
from wtforms import SubmitField, IntegerField
from wtforms.validators import DataRequired
from flask import render_template, flash, redirect
import numpy as np
import os

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = "12345678"
model = svm_model()

class config(object):
    KEY = os.environ.get("SECRET_KEY") or "12345678"

app.config.from_object(config)

class value_input(FlaskForm):
    pm10 = IntegerField("pm10")
    so2 = IntegerField("so2")
    o3 = IntegerField("o3")
    no2 = IntegerField("no2")
    submit = SubmitField("prediksi")

@app.route('/')
def form_predict():
    form = value_input()
    return render_template('main.html', title='deploy microdential 2', form=form)

@app.route('/', methods=['GET', "POST"])
def predict():
    form = value_input()
    if(form.validate_on_submit()):
        flash("kategori keadaannya {}".format( model.predict( np.array([[
            form.pm10.data, form.so2.data, form.o3.data, form.no2.data
        ]]) )))
        return redirect('/')
    return render_template('main.html', title="deploy microdential 2", form=form)

if(__name__ == "__main__"):
    app.run(debug=True)