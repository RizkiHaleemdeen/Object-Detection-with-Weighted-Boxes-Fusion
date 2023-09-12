FROM python:3.9-slim-buster
WORKDIR /Users/admin/Documents/Object Detection/Docker
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]