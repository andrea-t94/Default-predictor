FROM python:3.8.5

#install requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

#copy dataset
COPY dataset.csv ./dataset.csv

#copy scripts
COPY train.py ./train.py
COPY inference.py ./inference.py
COPY feature_config.py ./feature_config.py
COPY model_config.py ./model_config.py
COPY default_predictor.py ./default_predictor.py
COPY preprocessor.py ./preprocessor.py

#train and deploy first model
RUN mkdir -p artifacts/
RUN mkdir -p models/
RUN python3 train.py

#automatic inference
RUN python3 inference.py

#app settings
COPY app.py ./app.py
COPY wsgi.py ./wsgi.py
# Expose port
EXPOSE 5000
# Start the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]

