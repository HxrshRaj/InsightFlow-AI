FROM golang:1.22-alpine AS build
WORKDIR /src
COPY . .
RUN CGO_ENABLED=0 go build -o /out/serve ./cmd/serve

FROM gcr.io/distroless/static-debian12
COPY --from=build /out/serve /serve
COPY testdata/model_le.bin /model/model.bin
EXPOSE 8088
ENTRYPOINT ["/serve", "-model", "/model/model.bin", "-addr", ":8088"]
