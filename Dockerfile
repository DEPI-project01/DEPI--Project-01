FROM tensorflow/tensorflow:2.13.0-jupyter

# Set the working directory
WORKDIR /usr/src/app

# Copy your current directory contents into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt



# Expose ports for Jupyter Notebook, MLflow server, and Streamlit app
EXPOSE 8888 5000

# Command to run Jupyter and MLflow tracking server
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --allow-root --no-browser & \
                    streamlit run app.py --server.port 5000 "]
