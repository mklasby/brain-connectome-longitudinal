FROM ubuntu:jammy-20230804

ARG USERNAME=user
ARG WORKSPACE_DIR=/home/user/procan_connectome

SHELL ["/bin/bash", "-c"]

# Use a non-root user
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

# Create the user
RUN groupadd --gid $USER_GID ${USERNAME} \
    && useradd --uid $USER_UID --gid $USER_GID -m ${USERNAME}

RUN mkdir ${WORKSPACE_DIR}/ && \
    chown -R $USER_GID:$USER_UID ${WORKSPACE_DIR}

# Some development helpers and add user as a sudoer
RUN apt-get update \
    && apt-get install -y git ssh tmux vim curl htop sudo python3.10 python3.10-dev python3-pip -y

RUN echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# Setup our users env
USER ${USERNAME}
ENV WORKSPACE_DIR=${WORKSPACE_DIR} \
    PATH="/home/${USERNAME}/.local/bin:${PATH}" \
    NVIDIA_DRIVER_CAPABILITIES="all"

# Install rust as we need it for rpds-py
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

WORKDIR /home/${USERNAME}
COPY ./requirements.txt ./
RUN pip install --upgrade pip &&  yes | pip install -r requirements.txt
RUN echo "alias python=python3" >> ~/.bashrc && \
    echo "alias pip=pip3" >> ~/.bashrc
