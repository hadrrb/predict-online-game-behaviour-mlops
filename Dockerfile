FROM mageai/mageai:latest

WORKDIR /home/src

COPY game_behaviour/requirements.txt .

RUN pip install -r requirements.txt