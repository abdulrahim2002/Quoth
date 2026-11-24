# Clean Arch box for validating the Quoth AUR recipe.
# makepkg refuses to run as root, so build as an unprivileged user with passwordless
# pacman for dependency install. Usage (from the repo root):
#   docker build -t quoth-arch packaging/aur/.dockervalidate
#   docker run --rm -v "$PWD":/work:ro quoth-arch
FROM archlinux:latest

# Arch's login PATH. Docker's default puts /usr/sbin first, and python -m installer
# would then write #!/usr/sbin/python into the packaged launcher.
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/bin

RUN pacman -Syu --noconfirm --needed base-devel git namcap sudo xorg-server-xvfb \
 && useradd -m builder \
 && echo 'builder ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/builder \
 && chmod 0440 /etc/sudoers.d/builder

COPY run.sh /usr/local/bin/quoth-validate
RUN chmod +x /usr/local/bin/quoth-validate

USER builder
WORKDIR /home/builder
CMD ["quoth-validate"]
