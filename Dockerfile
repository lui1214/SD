FROM python:3.9
ADD . /app
WORKDIR /app
EXPOSE 8000
RUN pip install -r requirements.txt
CMD ["mercury", "run", "0.0.0.0:8000"]