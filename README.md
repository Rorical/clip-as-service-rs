# CLIP as service in Rust

A blazing fast gRPC server for CLIP model, powered by ONNX.

**Only one of text model or vision model can be loaded.**

## Build
```bash
cargo build --bin clip-as-service-server --release
```

## Use

### Text mode
At `https://github.com/jina-ai/clip-as-service/blob/main/server/clip_server/model/clip_onnx.py`, download one of the onnx text model and place it inside `data` folder along with the program.
Also go to huggingface and download corresponding tokenizer config `tokenizer.json` and place it inside `data` folder.

### Vision mode
place `vision.onnx` inside the data directory.

then run this program.

## Client

This is a gRPC service. Feel free to create your client implementation by simply just taking the `pb/encoder/encoder.proto` file and complie it using protobuf.
Golang client implementation is at [https://github.com/Rorical/clip-as-service-rs-go-cli](https://github.com/Rorical/clip-as-service-rs-go-cli)
