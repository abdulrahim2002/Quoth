# Clean Ubuntu 24.04 (noble) box for validating the Quoth .deb recipe. noble is
# the hardest target: it's Pop!_OS's base and ships no system gtk4-layer-shell and
# numpy 1.26, so a green run here proves the bundled gtk4-layer-shell + OpenVINO and
# the low dependency floors. Usage (from the repo root):
#   docker build -t quoth-deb packaging/debian/.dockervalidate
#   docker run --rm -v "$PWD":/work:ro quoth-deb
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential debhelper dpkg-dev \
      ca-certificates curl git lintian python3 xvfb xauth \
 && rm -rf /var/lib/apt/lists/*

COPY run.sh /usr/local/bin/quoth-validate
RUN chmod +x /usr/local/bin/quoth-validate

WORKDIR /root
CMD ["quoth-validate"]
