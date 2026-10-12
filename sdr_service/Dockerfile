FROM docker.io/springepfl/synthetic-data:latest

RUN python3 -m pip install --no-cache-dir fastapi "uvicorn[standard]"

WORKDIR /sdr_service
COPY app.py /sdr_service/app.py

EXPOSE 8000

ENTRYPOINT []
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
