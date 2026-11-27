FROM docker.io/springepfl/synthetic-data:latest

LABEL org.opencontainers.image.source=https://github.com/SCAI-BIO/guided-sdg
LABEL org.opencontainers.image.licenses=GPL-3.0-or-later

RUN python3 -m pip install --no-cache-dir fastapi "uvicorn[standard]"

WORKDIR /sdr_service
COPY app.py /sdr_service/app.py

EXPOSE 8000

ENTRYPOINT []
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
