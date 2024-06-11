FROM python:3.9

# 
WORKDIR /code

# Copy necessary files to the container
COPY requirements.txt .
COPY main.py .
COPY train_utils.py .
COPY utils.py .
COPY config.json .

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

ENTRYPOINT [ "python" ]

# Run main.py when the container launches
CMD [ "main.py" ]