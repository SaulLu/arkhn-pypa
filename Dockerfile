# alpine is a lightweight linux distro
FROM python:3.7-alpine
COPY . /src
WORKDIR /src
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
