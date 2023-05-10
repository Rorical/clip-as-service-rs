use std::env;
use std::io::Cursor;
use std::ops::Index;
use std::path::Path;
use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status};
pub mod encoder {
    tonic::include_proto!("encoder");
}
use encoder::{ EncodeTextRequest, EncoderResponse, Embedding };
use encoder::encoder_server::{ EncoderServer, Encoder };

use tokenizers::tokenizer::{Tokenizer};
use ndarray::{Array, Array2, Array3, Array4, ArrayBase, ArrayD, ArrayView3, Axis, Dim, OwnedRepr, s, stack, ViewRepr};
use ort::session::Session;
use ort::{
    tensor::{DynOrtTensor, FromArray, InputTensor, OrtOwnedTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder
};

use itertools::Itertools;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy};
extern crate image;
use image::io::Reader as ImageReader;

use clap::Parser;
use image::{GenericImageView, Pixel};
use image::imageops::FilterType;
use crate::encoder::EncodeImageRequest;

extern crate num_cpus;

#[derive(Parser, Debug)]
#[command(author = "Ro <rorical@shugetsu.space>", version = "0.1", about = "Clip service", long_about = None)]
struct Args {
    /// Address to listen
    #[arg(short, long, default_value = "[::1]:50051")]
    listen: String,

    /// Model type, default text
    #[arg(short, long, default_value_t = false)]
    vision_mode: bool,

    /// Vision model input image size, default 224
    #[arg(short, long, default_value_t = 224)]
    vision_size: usize,
}


pub struct EncoderService {
    tokenizer: Tokenizer,
    encoder: Session,
    vision_mode: bool,
    vision_size: usize,
}

impl EncoderService {
    fn new(environment: &Arc<Environment>, args: &Args) -> EncoderService {
        let vision_mode = args.vision_mode;

        let model_path = if vision_mode {
            "vision.onnx"
        } else {
            "textual.onnx"
        };
        let tokenizer_path = "tokenizer.json";

        let root = Path::new("data/");
        assert!(env::set_current_dir(&root).is_ok());

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
        let encoder = SessionBuilder::new(environment).unwrap()
            .with_inter_threads(num_cpus as i16).unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level2).unwrap()
            .with_model_from_file(model_path).unwrap();

        EncoderService {
            tokenizer,
            encoder,
            vision_mode: vision_mode,
            vision_size: args.vision_size,
        }
    }

    pub fn _process_text(&self, text: &Vec<String>) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        let preprocessed = self.tokenizer.encode_batch(text.clone(), true)?;
        let v1: Vec<i32> = preprocessed.iter().map(|i| i.get_ids().iter().map(|b| *b as i32).collect()).concat();
        let v2: Vec<i32> = preprocessed.iter().map(|i| i.get_attention_mask().iter().map(|b| *b as i32).collect()).concat();

        let ids = Array2::from_shape_vec((text.len(), v1.len()/text.len()), v1).unwrap();
        let mask = Array2::from_shape_vec((text.len(), v2.len()/text.len()), v2).unwrap();

        let outputs= self.encoder.run([InputTensor::from_array(ids.into_dyn()), InputTensor::from_array(mask.into_dyn())]).unwrap();
        let binding = outputs[0].try_extract().unwrap();
        let embeddings = binding.view();

        let seq_len = embeddings.shape().get(1).unwrap();

        Ok(embeddings.iter().map(|s| *s).chunks(*seq_len).into_iter().map(|b| b.collect()).collect())
    }

    pub fn _process_image(&self, images_bytes: &Vec<Vec<u8>>) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        let mean = vec![0.48145466, 0.4578275, 0.40821073]; // CLIP Dataset
        let std = vec![0.26862954, 0.26130258, 0.27577711];

        let mut pixels = Array4::<f32>::zeros(Dim([images_bytes.len(), 3, self.vision_size, self.vision_size]));
        for (index, image_bytes) in images_bytes.iter().enumerate() {
            let image = ImageReader::new(Cursor::new(image_bytes)).with_guessed_format()?.decode()?;
            let image = image.resize_exact(self.vision_size as u32, self.vision_size as u32, FilterType::CatmullRom);
            for (x, y, pixel) in image.pixels() {
                pixels[[index, 0, x as usize, y as usize]] = (pixel.0[0] as f32 / 255.0 - mean[0]) / std[0];
                pixels[[index, 1, x as usize, y as usize]] = (pixel.0[1] as f32 / 255.0 - mean[1]) / std[1];
                pixels[[index, 2, x as usize, y as usize]] = (pixel.0[2] as f32 / 255.0 - mean[2]) / std[2];
            }
        }

        let outputs= self.encoder.run([InputTensor::from_array(pixels.into_dyn())]).unwrap();
        let binding = outputs[0].try_extract().unwrap();
        let embeddings = binding.view();

        let seq_len = embeddings.shape().get(1).unwrap();

        Ok(embeddings.iter().map(|s| *s).chunks(*seq_len).into_iter().map(|b| b.collect()).collect())
    }
}

#[tonic::async_trait]
impl Encoder for EncoderService {
    async fn encode_text(&self, request: Request<EncodeTextRequest>) -> Result<Response<EncoderResponse>, Status> {
        if self.vision_mode {
            return Err(Status::invalid_argument("wrong model is loaded"))
        }
        let texts = &request.get_ref().texts;
        let embedding = self._process_text(texts).unwrap().into_iter().map(
            |i| Embedding {
                point: i,
            }
        ).collect();
        Ok(Response::new(EncoderResponse {
            embedding,
        }))
    }
    async fn encode_image(&self, request: Request<EncodeImageRequest>) -> Result<Response<EncoderResponse>, Status> {
        if !self.vision_mode {
            return Err(Status::invalid_argument("wrong model is loaded"))
        }
        let images = &request.get_ref().images;
        let embedding = self._process_image(images).unwrap().into_iter().map(
            |i| Embedding {
                point: i,
            }
        ).collect();
        Ok(Response::new(EncoderResponse {
            embedding
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let addr: &String = &args.listen;
    let environment =
        Arc::new(Environment::builder()
            .with_name("clip")
            .build().unwrap());

    let server = EncoderService::new(&environment, &args);

    println!("Listening at {:?} with {} mode.", addr, if args.vision_mode {
        "vision"
    } else {
        "text"
    });
    Server::builder()
        .add_service(EncoderServer::new(server))
        .serve(addr.parse()?)
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::io::prelude::*;
    use std::fs::File;
    use std::sync::Arc;
    use ort::environment::Environment;
    use crate::{Args, EncoderService};

    #[test]
    fn it_works() {
        let environment =
            Arc::new(Environment::builder()
                .with_name("clip")
                .build().unwrap());
        let args = Args{
            listen: "".to_string(),
            vision_mode: true,
            vision_size: 224,
        };
        let service = EncoderService::new(&environment, &args);

        let mut f = File::open("test.jpg").unwrap();
        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer).unwrap();

        let images = vec![buffer];
        let emb = service._process_image(&images).unwrap();
        println!("{:?}", emb[0].len());
    }
}