use std::cell::{Cell, RefCell};
use std::ops::Deref;
use tonic::{transport::Server, Request, Response, Status};
pub mod encoder {
    tonic::include_proto!("encoder");
}
use encoder::{ EncodeTextRequest, EncodeTextResponse, Embedding };
use encoder::encoder_server::{ EncoderServer, Encoder };

use tokenizers::tokenizer::{Tokenizer};
use onnxruntime::environment::Environment;
use onnxruntime::GraphOptimizationLevel;
use onnxruntime::ndarray::{Array2, Axis};
use onnxruntime::session::Session;
use onnxruntime::tensor::OrtOwnedTensor;

use itertools::Itertools;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, tokenizer};

mod utils;
use crate::utils::unsafe_sync::UnsafeSendSync;

extern crate num_cpus;

pub struct EncoderService {
    tokenizer: Tokenizer,
    encoder: UnsafeSendSync<RefCell<Session<'static>>>,
}

impl EncoderService {
    fn new(environment: &'static Environment) -> EncoderService {
        let model_path = "data/textual.onnx";
        let tokenizer_path = "data/tokenizer.json";

        let mut tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        tokenizer.with_padding(Option::from(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string()
        }));

        let num_cpus = num_cpus::get();
        let mut encoder: Session  = environment
            .new_session_builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::All).unwrap()
            .with_number_threads(num_cpus as i16).unwrap()
            .with_model_from_file(model_path).unwrap();

        let encoder = UnsafeSendSync::new(RefCell::new(encoder));
        EncoderService {
            tokenizer,
            encoder,
        }
    }

    pub fn _process_text(&self, text: &Vec<String>) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        let preprocessed = self.tokenizer.encode_batch(text.clone(), true)?;
        let v1: Vec<i32> = preprocessed.iter().map(|i| i.get_ids().iter().map(|b| *b as i32).collect()).concat();
        let v2: Vec<i32> = preprocessed.iter().map(|i| i.get_attention_mask().iter().map(|b| *b as i32).collect()).concat();

        let ids = Array2::from_shape_vec((text.len(), v1.len()/text.len()), v1).unwrap();
        let mask = Array2::from_shape_vec((text.len(), v2.len()/text.len()), v2).unwrap();

        let inputs = vec![ids, mask];
        let mut encoder = self.encoder.borrow_mut();
        let outputs: Vec<OrtOwnedTensor<f32, _>> = encoder.run(inputs).unwrap();
        let seq_len = *outputs[0].shape().get(1).unwrap();

        Ok(outputs[0].iter().map(|s| *s).chunks(seq_len).into_iter().map(|b| b.collect()).collect())
    }
}

#[tonic::async_trait]
impl Encoder for EncoderService {
    async fn encode_text(&self, request: Request<EncodeTextRequest>) -> Result<Response<EncodeTextResponse>, Status> {
        let texts = &request.get_ref().texts;
        let embedding = self._process_text(texts).unwrap().into_iter().map(
            |i| Embedding {
                point: i,
            }
        ).collect();
        Ok(Response::new(EncodeTextResponse {
            embedding,
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let environment =
        Box::leak(Box::new(Environment::builder()
            .with_name("clip")
            .build().unwrap()));

    let server = EncoderService::new(environment);

    println!("Listening at {:?}", addr);
    Server::builder()
        .add_service(EncoderServer::new(server))
        .serve(addr)
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use onnxruntime::environment::Environment;
    use crate::EncoderService;

    #[test]
    fn it_works() {
        let environment =
        Box::leak(Box::new(Environment::builder()
            .with_name("clip")
            .build().unwrap()));
        let mut service = EncoderService::new(environment);
        let texts = vec!["你好".to_string(), "hello".to_string()];
        let emb = service._process_text(&texts).unwrap();
        println!("{:?}", emb.len());
    }
}