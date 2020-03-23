# Build image
# >>> docker build -t wdobbels/morpho-ml .

# Run by binding the project directory to /morphoml
# Run as root. Adding -u $(id -u):$(id -g) makes it run as current user, but seems to
# lack necessary privileges. However, running as root is not recommended by
# tensorflow (just start with 'bash' command to see warning).
# >>> docker run -it --name morphoml -v "$(pwd)":/morphoml \
#       -v "${HOME}/.jupyter:/root/.jupyter" \
#       -p 8989:8989 wdobbels/morpho-ml

FROM tensorflow/tensorflow:2.1.0-gpu-py3
RUN apt-get update && apt-get install -y python3-pip && \
    curl -sL https://deb.nodesource.com/setup_13.x | bash - && \
    apt-get install -y nodejs
ENV JUPYTERLAB_DIR .jupyter-applications
RUN python3 -m pip install numpy scipy matplotlib pandas scikit-learn h5py \
    pillow tensorflow-addons jupyterlab ipywidgets wdplot \
    && jupyter nbextension enable --py widgetsnbextension \
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager
EXPOSE 8989
WORKDIR /morphoml
CMD jupyter-lab --no-browser --port=8989 --ip=0.0.0.0 --allow-root