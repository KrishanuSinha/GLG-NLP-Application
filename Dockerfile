FROM python:3.8

WORKDIR /KRISHANU_NLP_APP

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY . /KRISHANU_NLP_APP

EXPOSE 8501

ENTRYPOINT [ "streamlit","run" ]

CMD [ "Deployed_NLP_Model.py" ]
