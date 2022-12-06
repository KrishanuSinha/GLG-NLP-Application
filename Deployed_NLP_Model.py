import streamlit as st
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import tweetnlp
import operator
import plotly.express as px
from huggingface_hub import create_repo
from PIL import Image
from streamlit_option_menu import option_menu
from pathlib import Path
import spacy_streamlit
import spacy
nlp = spacy.load("en_core_web_sm")


st.set_page_config(page_title="De-mystify Customer's Requests",
                       page_icon=":wave:",
                       layout="wide")


#horizontal menu
selected = option_menu(None, ["My NLP App", "About Me"], 
    icons=['bar-chart-line-fill', 'emoji-smile-fill'], 
    menu_icon="cast", default_index=0, orientation="horizontal")


if selected == "My NLP App":
    

    st.subheader("Let's simplify texts")
    #st.title(" :hotsprings:De-mystify Customer Requests:hotsprings: ")
    #st.markdown("##")
    #@st.cache(persist=True ,allow_output_mutation=False)


    def get_model():
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = tweetnlp.load_model('topic_classification')
        return tokenizer,model

    user_input = st.text_area('Enter Text to Analyze','Enter text here')
    #button = st.button("Analyze")

    tokenizer,model = get_model()

    #if user_input and button :
        # WOULD THE USER LIKE TO APPLY TOPIC MODELLING ON THE TEXT
        #topic_model = st.checkbox("Find Topics")
    if st.checkbox("Find Topics"):
        test_sample = [user_input]
        # Note: the probability of the multi-label model is the output of sigmoid function on binary prediction whether each topic is positive or negative.
        #model = tweetnlp.load_model('topic_classification')  # Or `model = tweetnlp.TopicClassification()`
        d = model.topic(test_sample, return_probability=True)
        d = d[0]
        lst = ', '.join(d['label'])
        st.write("The most likely category is: ",lst)
        dict_vals = d['probability']
        sorted_d = dict( sorted(dict_vals.items(), key=operator.itemgetter(1),reverse=False))
        df = pd.DataFrame.from_dict(sorted_d.items())
        df = df.rename({0: 'Topics', 1: 'Probability'}, axis='columns')
        df['Probability'] = df['Probability'].round(decimals = 2)
        
        #st.write(df)
        #names = list(sorted_d.keys())
        #values = list(sorted_d.values())
        #df = px.data.tips()
        #st.write(f"It most likely falls under the {lst} category")
    

        fig = px.bar(df, 
                         x="Probability", 
                         y="Topics", 
                         orientation="h",
                         title="<b>Topic likelyhood</b>",
                         text="Probability",
                         color_discrete_sequence=["goldenrod"],
                         template="plotly_white",
                         hover_data=['Probability', 'Topics'], 
                         labels={'Probability':'Probability of Topic'}, height=400)

        fig.update_layout(
                    font=dict(
                        family="Courier New, monospace",
                        size=18,  # Set the font size here
                        color="White")
                          )
                 
        st.plotly_chart(fig)

    # WOULD THE USER LIKE TO APPLY NER ON THE TEXT    
    ner_model = st.checkbox("Apply NER")
    if ner_model:
        # DISPLAY NER MODEL OUTPUT
        docx = nlp(test_sample[0])
        spacy_streamlit.visualize_ner(docx, labels=nlp.get_pipe("ner").labels)

        # WOULD THE USER LIKE TO SUMMARIZE THE TEXT
    summaryzer_model = st.checkbox("Apply Summarize")
    if summaryzer_model:
        # DISPLAY TEXT SUMMARYZER MODEL OUTPUT
        st.header("Text Summarizer")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        st.info("Used Bart-CNN, fine-tuned on CNN Daily Mail to summarize text ")
        #st.text("Used Bart-CNN, fine-tuned on CNN Daily Mail")
        summary_result = summarizer(test_sample[0], max_length=130, min_length=30, do_sample=False)
        st.write(summary_result[0]["summary_text"])
    
    st.sidebar.subheader("References:")
    st.sidebar.text("1. Topic modeling: David M. Blei, Andrew Y. Ng, Michael I. Jordan , Latent Dirichlet Allocation , Journal of Machine Learning Research 3 (2003) 993-1022")
    st.sidebar.text("Link: https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf")
    st.sidebar.text("2. Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Céspedes, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, Ray Kurzweil, Universal Sentence Encoder, arXiv:1803.11175, 2018.")
    st.sidebar.text("NER Entity:")
    st.sidebar.text("1. Link: https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder")
    st.sidebar.text("Summaryzation:") 
    st.sidebar.text("https://huggingface.co/facebook/bart-large-cnn")
    st.sidebar.info("Also, Cudos to the FourthBrain Team 🙌")
	

    st.sidebar.subheader("Author")
    st.sidebar.text("Krishanu Prabha Sinha")
    st.sidebar.text("krishanusinha12@gmail.com")


if selected == "About Me":


    # --- PATH SETTINGS ---
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    css_file = current_dir / "styles" / "main.css"
    resume_file = current_dir / "assets" / "Krishanu_prabha_Sinha_Resume8.pdf"
    profile_pic = current_dir / "assets" / "My_Image.png"


    # --- GENERAL SETTINGS ---
    PAGE_TITLE = "Digital CV | Krishanu Prabha Sinha"
    PAGE_ICON = ":wave:"
    NAME = "Krishanu Prabha Sinha"
    PRONOUNS = "He/Him/His"
    EMAIL = "krishanusinha12@gmail.com"
    DESCRIPTION = """
    Experienced Machine Learning Engineer with strong AWS Cloud and interpersonal skills. Seeking to be part of the revolution that is going to redefine how companies leverage data
    """
    SOCIAL_MEDIA = {
        "LinkedIn": "https://www.linkedin.com/in/krishanu-prabha-sinha/",
        "GitHub": "https://github.com/KrishanuSinha",
        "AI-Publications": "https://ieeexplore.ieee.org/document/9599133",
        "Data Science Blogs": "https://dataenthusiast863890119.wordpress.com/"
    }
    PROJECTS = {
        "🏆 DETERMING FACTORS IMPACTING THE RETENTION RATE IN US UNIVERSITIES VIA ENSEMBLE ML ALGORITHMS": "https://dataenthusiast863890119.wordpress.com/2018/06/14/determing-factors-impacting-the-retention-rate-in-us-universities-via-ensemble-ml-algorithms/",
        "🏆 CLASSIFYING YOUTUBE VIDEOS BASED ON TEXTUAL DATA AND STATISTICS": "https://dataenthusiast863890119.wordpress.com/2018/08/18/classifying-youtube-videos-based-on-textual-data-and-statistics/"
    }
    PUBLICATIONS = {
        "🏆 Agentless Insurance Model Based on Modern Artificial Intelligence": "https://ieeexplore.ieee.org/document/9599133",
    }


    #st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)


    # --- LOAD CSS, PDF & PROFIL PIC ---
    with open(css_file) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    with open(resume_file, "rb") as pdf_file:
        PDFbyte = pdf_file.read()
    profile_pic = Image.open(profile_pic)


    # --- HERO SECTION ---
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.image(profile_pic, width=490)

    with col2:
        st.title(NAME)
        st.subheader(PRONOUNS)
        st.subheader(EMAIL)
        st.write(DESCRIPTION)
        st.download_button(
            label=" 📄 Download Resume",
            data=PDFbyte,
            file_name=resume_file.name,
            mime="application/octet-stream",
            )
        


    # --- SOCIAL LINKS ---
    st.write('\n')
    cols = st.columns(len(SOCIAL_MEDIA))
    for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
        cols[index].write(f"[{platform}]({link})")


    # --- EXPERIENCE & QUALIFICATIONS ---
    st.write('\n')
    st.subheader("Experience & Qualifications")
    st.write(
        """
    - ✔️ 6+ years of experience in developing and deploying Data Science projects with customer-facing roles to improve business processes within AGILE & SDLC for business operations through its analytics arm.
    - ✔️ Strong hands on experience and knowledge in Python and AWS Cloud Services
    - ✔️ Good understanding of statistical principles and their respective applications
    - ✔️ Excellent team-player and displaying strong sense of initiative on tasks
        """
            )


    # --- SKILLS ---
    st.write('\n')
    st.subheader("Hard Skills")
    st.write(
        """
        - ✔️ DATA SCIENCE SKILLS: Machine Learning, Data Collection, Python, Data Preprocessing, Natural Language Processing, Data Visualizing, Recommendation System, Neural Networks, Time Series Analysis, Flask Model Deployment, AWS Sagemaker.
        - ✔️ DATA VISUALIZATION TOOLS: Tableau, Matplotlib, Seaborn.
        - ✔️ PYTHON WEB DEVELOPMENT: grequests, urllib, urllib2, Beautifulsoup, celery, pickle, jinja2, itertools, lxml, matplotlib, Pandas, Numpy, Javascript, React.
        - ✔️ AWS Services : Amazon S3, EC2, RDS, IAM, Elastic Load Balancing,Auto Scaling, Cloudwatch, SNS, SQS, SES, Lambda, EMR, Redshift, Boto3, DynamoDB, Terraform.

        """
            )


    # --- WORK HISTORY ---
    st.write('\n')
    st.subheader("Work History")
    st.write("---")

    # --- JOB 1
    st.write("💻", "**Machine Learning Engineer | StateFarm**")
    st.write("September 2018 - Current")
    st.write(
        """
    - ► Developed and deployed in AWS a web based data science tool using Python, Flask, HTML/CSS, Javascript ,React that guides user through generic feature identification processes. This tool automates data summarizing, cleaning, visualization, and analysis and outputs data features impacting target and confidence from those features. Users without machine learning coding experience (Python/R/SAS) can benefit without the need for a dedicated Data Scientist.
    - ► Built and deployed ML Model that was able to classify insurance claims that were Closed with Payment and Without Payment in AWS Sage maker.
    - ► Implemented AWS Step functions to automate and orchestrate the Amazon SageMaker related tasks such as publishing data into S3, training ML model and deploying it for prediction.
    - ► Performed data cleaning EDA using seaborn on the data to find useful insights, outliers, and missing values on structured and unstructured data.
    - ► Build a predictive model to predict the gender of candidates using LSTM and Skip - Gram models. Achieved an accuracy of 85% on the initial test dataset.
    - ► Sentiment Analysis of Statefarm Insurance Blogs using NLP techniques like GloveVecs, Doc2Vecs , LSTMs, LDAs, CNN to classify readable from non-readable.
        """
            )

    # --- JOB 2
    st.write('\n')
    st.write("💻", "**Data Engineer | Samsung Research Institute, India**")
    st.write("10/2012 - 09/2016")
    st.write(
        """
    - ► Implemented cloud policies, managed technology requests and maintained service availability.
    - ► Assessed organization technology and managed cloud migration process.
    - ► Created Lambda functions with Boto3 to perform various data enrichment on ingested data.
    - ► Extracted data using SQL queries in Tableau to design a Dashboard for insights and Key Trends.
    - ► Analyzed Large Business Data-sets to provide Strategic Direction to the company using Data Analytics.
    - ► Used Statistical Techniques for Hypothesis Testing to validate Data and Interpretations.
    - ► Presented Findings and Data to the team to improve Strategies and Operations.
    - ► Gained a good understanding of Hadoop ecosystem tools like apache spark.
    - ► Created Tableau Scorecards, Dashboards and other data visualizations using tableau desktop. technical environment: RDBMS, SQL, Python, Tableau, Pyspark.
        """
            )




    # --- Projects & Accomplishments ---
    # --- SKILLS ---
    st.write('\n')
    st.subheader("Hard Skills")
    st.write(
        """
        - ✔️ DATA SCIENCE SKILLS: Machine Learning, Data Collection, Python, Data Preprocessing, Natural Language Processing, Data Visualizing, Recommendation System, Neural Networks, Time Series Analysis, Flask Model Deployment, AWS Sagemaker.
        - ✔️ DATA VISUALIZATION TOOLS: Tableau, Matplotlib, Seaborn.
        - ✔️ PYTHON WEB DEVELOPMENT: grequests, urllib, urllib2, Beautifulsoup, celery, pickle, jinja2, itertools, lxml, matplotlib, Pandas, Numpy, Javascript, React.
        - ✔️ AWS Services : Amazon S3, EC2, RDS, IAM, Elastic Load Balancing,Auto Scaling, Cloudwatch, SNS, SQS, SES, Lambda, EMR, Redshift, Boto3, DynamoDB, Terraform.

        """
            )


    # --- EDUCATION HISTORY ---
    st.write('\n')
    st.subheader("Education:")
    st.write("---")

    # --- MASTERS
    st.write("🎓", "**University of Houston at Clear Lake**")
    st.write("Masters in Data Analytics")
    

    # --- BACHELORS
    st.write('\n')
    st.write("🎓", "**Visveswaraya Technological University**")
    st.write("Bachelor in Computer Science")
    
    #with header:
    #    st.title("GLG Topic Modeling Project:")

    #with dataset:
    #    st.title("GLG Dataset:")
    #    df = pd.read_csv("/home/krishanu/MLE-9/MLE-9/assignments/Streamlit/GLG_Final_Dataset.csv")
    #    st.write(df.head(20))
    #    st.subheader('Section Distribution')
    #    section_count = pd.DataFrame(df['section'].value_counts())
    #    st.bar_chart(section_count)
    

    #with features:
    #    st.title("GLG Features:")
    #    st.markdown("* **first feature:** I created this because of this ")
    #    st.markdown("* **first feature:** I created that because of this ")

    #with model_training:
    #    st.title("GLG Topic Model Training:")
    #    sel_col, disp_col = st.columns(2)

    #    max_depth = sel_col.slider("What should be the max depth of the model?", min_value=10, max_value=100, value=20, step=10)
    #    n_estimators = sel_col.selectbox("How many trees should be there?", options=[100,200,300,'No Limit'], index =0)
    #    input_feature = sel_col.text_input('Which feature should be used as the input features?','section_count')
