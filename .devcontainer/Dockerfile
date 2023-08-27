# Basic Python and Node.js image information
ARG VARIANT="3.11-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}
ARG NODE_VERSION="none"

RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

#Install poetry official way
USER vscode
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH = "${PATH}:/home/vscode/.local/bin"

## Git setup
RUN git config --global diff.tool vscode && git config --global difftool.vscode.cmd 'code --wait --diff $LOCAL $REMOTE'