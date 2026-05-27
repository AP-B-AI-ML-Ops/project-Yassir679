FROM python:3.12-bullseye

RUN pip install -U pip

COPY . /

RUN pip install -r ./requirements.txt --upgrade

EXPOSE 8000

CMD ["python", "app.py"]
