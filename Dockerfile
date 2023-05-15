# syntax=docker/dockerfile:1.4
FROM --platform=$BUILDPLATFORM python:3.10 AS builder

WORKDIR /explainable-tar

COPY ./explainable-tar/requirements.txt ./explainable-tar/requirements.txt
RUN  pip3 install -r explainable-tar/requirements.txt

COPY ./explainable-tar /explainable-tar
COPY ./data/reuters21578 ../data/reuters21578
COPY ./data/pickels/reuters_tokens.pkl ../data/pickels/reuters_tokens.pkl

ENTRYPOINT ["python3"]
CMD ["app.py"]

FROM builder as dev-envs

RUN <<EOF
apk update
apk add git
EOF

RUN <<EOF
addgroup -S docker
adduser -S --shell /bin/bash --ingroup docker vscode
EOF
# install Docker tools (cli, buildx, compose)
COPY --from=gloursdocker/docker / /