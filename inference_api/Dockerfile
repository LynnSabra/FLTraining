FROM fl
COPY . .
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential
RUN pip3 install fastapi[all]
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD ["uvicorn", "start:app", "--host", "0.0.0.0", "--port", "5555"]
