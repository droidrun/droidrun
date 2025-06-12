# Credits to https://github.com/amrsa1/Android-Emulator-image

FROM openjdk:18-jdk-slim AS androidsdk

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /
#=============================
# Install Dependenices
#=============================
SHELL ["/bin/bash", "-c"]
RUN apt update && apt install -y curl sudo wget unzip bzip2 libdrm-dev libxkbcommon-dev libgbm-dev libasound-dev libnss3 libxcursor1 libpulse-dev libxshmfence-dev xauth xvfb x11vnc fluxbox wmctrl libdbus-glib-1-2 ffmpeg

#==============================
# Android SDK ARGS
#==============================
ARG ARCH="x86_64"
ARG TARGET="google_apis" 
ARG API_LEVEL="33"
ARG BUILD_TOOLS="33.0.0"
#ARG ANDROID_ARCH=${ANDROID_ARCH_DEFAULT}
ARG ANDROID_API_LEVEL="android-${API_LEVEL}"
ARG ANDROID_APIS="${TARGET};${ARCH}"
ARG EMULATOR_PACKAGE="system-images;${ANDROID_API_LEVEL};${ANDROID_APIS}"
ARG PLATFORM_VERSION="platforms;${ANDROID_API_LEVEL}"
ARG BUILD_TOOL="build-tools;${BUILD_TOOLS}"
ARG ANDROID_CMD="commandlinetools-linux-11076708_latest.zip"
ARG ANDROID_SDK_PACKAGES="${EMULATOR_PACKAGE} ${PLATFORM_VERSION} ${BUILD_TOOL} platform-tools emulator"

#==============================
# Set JAVA_HOME - SDK
#==============================
ENV ANDROID_SDK_ROOT=/opt/android
ENV PATH="$PATH:$ANDROID_SDK_ROOT/cmdline-tools/tools:$ANDROID_SDK_ROOT/cmdline-tools/tools/bin:$ANDROID_SDK_ROOT/emulator:$ANDROID_SDK_ROOT/tools/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/build-tools/${BUILD_TOOLS}"
ENV DOCKER="true"

#============================================
# Install required Android CMD-line tools
#============================================
RUN wget https://dl.google.com/android/repository/${ANDROID_CMD} -P /tmp && \
    unzip -d $ANDROID_SDK_ROOT /tmp/$ANDROID_CMD && \
    mkdir -p $ANDROID_SDK_ROOT/cmdline-tools/tools && cd $ANDROID_SDK_ROOT/cmdline-tools &&  mv NOTICE.txt source.properties bin lib tools/  && \
    cd $ANDROID_SDK_ROOT/cmdline-tools/tools && ls

#============================================
# Install required package using SDK manager
#============================================
RUN yes Y | sdkmanager --licenses
RUN yes Y | sdkmanager --verbose --no_https ${ANDROID_SDK_PACKAGES}
#============================================
# Create required emulator
#============================================
ARG EMULATOR_NAME="Pixel_6_API_33"
ARG EMULATOR_DEVICE="pixel_6"
ENV EMULATOR_NAME=$EMULATOR_NAME
ENV DEVICE_NAME=$EMULATOR_DEVICE
RUN echo "no" | avdmanager --verbose create avd --force --name "${EMULATOR_NAME}" --device "${EMULATOR_DEVICE}" --package "${EMULATOR_PACKAGE}"

FROM androidsdk AS deps

#====================================
# Install Python 3.12.11 from source
#====================================
RUN apt-get update && \
    apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
    libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev git && \
    wget https://www.python.org/ftp/python/3.12.11/Python-3.12.11.tgz && \
    tar -xvf Python-3.12.11.tgz && \
    cd Python-3.12.11 && \
    ./configure --enable-optimizations && \
    make -j $(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.12.11* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create symlinks for python3.12
RUN ln -sf /usr/local/bin/python3.12 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python3.12 /usr/local/bin/python

RUN wget https://bootstrap.pypa.io/get-pip.py && python3.12 get-pip.py && rm get-pip.py

WORKDIR /opt/shared

# clone & patch android world
RUN git clone https://github.com/google-research/android_world.git && \
    find /opt/shared/android_world/android_world -type d -exec bash -c 'f="$0/__init__.py"; [ -f "$f" ] || touch "$f"' {} \; && \
    sed -i 's/^packages = /# packages = /' /opt/shared/android_world/pyproject.toml && \
    sed -i 's/^android_world = /# android_world = /' /opt/shared/android_world/pyproject.toml && \
    sed -i "s/'proto\/\*\.proto'/'proto\/\*\.proto', 'proto\/\*\.textproto'/" /opt/shared/android_world/setup.py

RUN cd /opt/shared/android_world && \
    pip install -r requirements.txt && \
    python setup.py install

ENV ANDROID_WORLD_PATH=/opt/shared/android_world/android_world

COPY . /opt/shared/droidrun
RUN pip install /opt/shared/droidrun

FROM deps AS benchmark

WORKDIR /opt/shared

#=========================
# Copying Scripts to root
#=========================
COPY ./eval/docker .

RUN chmod a+x start_emu_headless.sh && \
    chmod a+x entrypoint.sh

ENTRYPOINT [ "./entrypoint.sh" ]
