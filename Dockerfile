FROM python:3.7-alpine

WORKDIR /srv

COPY requirements.txt /srv/requirements.txt
RUN pip install -r requirements.txt

COPY ./src /srv

CMD ["python", "main.py"]

