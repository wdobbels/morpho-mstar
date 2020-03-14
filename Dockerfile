# Run by binding the project directory to /morphoml
# docker build .
# docker run -it --name morphoml -v "$(pwd)":/morphoml
FROM tensorflow/tensorflow:latest-gpu-jupyter
# Install more needed stuff here if needed
EXPOSE 8989
WORKDIR /morphoml
CMD jupyter-lab --no-browser --port=8989