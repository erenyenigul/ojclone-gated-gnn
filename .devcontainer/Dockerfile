# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.0/containers/ubuntu/.devcontainer/base.Dockerfile

# [Choice] Ubuntu version (use ubuntu-22.04 or ubuntu-18.04 on local arm64/Apple Silicon): ubuntu-22.04, ubuntu-20.04, ubuntu-18.04
ARG VARIANT="jammy"
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

RUN sudo apt-get update

RUN sudo apt-get install cmake

RUN sudo apt install -y graphviz libgraphviz-dev
RUN sudo apt install -y libllvm10 llvm-10-dev
RUN sudo apt install -y clang-10 libclang1-10 libclang-10-dev libclang-common-10-dev

RUN pip install -f data.pyg.org/whl/torch-1.13.0+cu116.html torch-sparse
RUN pip install -f data.pyg.org/whl/torch-1.13.0+cu116.html torch-scatter

RUN sudo apt install gcc-10 g++-10 