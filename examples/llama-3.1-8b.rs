use tenten::Tensor;

struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

struct RotaryEmbedding {}

struct MLP {}

struct Attention {}

#[derive(Default)]
struct DecoderLayer;

#[derive(Default)]
struct LlamaModel {
    decoder_layers: Vec<DecoderLayer>,
}

fn main() {
    let mut model: LlamaModel = Default::default();

    model.decoder_layers
}
