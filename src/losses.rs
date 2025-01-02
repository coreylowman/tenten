use crate::tensor::*;

pub fn mse(x: Tensor, target: Tensor) -> Result<Tensor, Error> {
    x.sub(target)?.square()?.mean()
}

pub fn mae(x: Tensor, target: Tensor) -> Result<Tensor, Error> {
    x.sub(target)?.abs()?.mean()
}

pub fn huber<S: Into<Scalar>>(x: Tensor, target: Tensor, delta: S) -> Result<Tensor, Error> {
    let dtype = x.dtype();
    let delta = Into::<Scalar>::into(delta).to_dtype(dtype);
    let diff = x.sub(target)?;
    let a = diff.clone().square()?.mul_scalar(0.5f64)?;
    let b = diff
        .clone()
        .abs()?
        .sub_scalar(delta / Into::<Scalar>::into(2.0f64).to_dtype(dtype))?
        .mul_scalar(delta)?;
    diff.lt_scalar(delta)?.choose(a, b)?.mean()
}

pub fn smooth_l1<S: Into<Scalar>>(x: Tensor, target: Tensor, delta: S) -> Result<Tensor, Error> {
    let delta = Into::<Scalar>::into(delta).to_dtype(x.dtype());
    huber(x, target, delta)?.div_scalar(delta)
}

pub fn kldiv_with_logits(logits: Tensor, target_probs: Tensor) -> Result<Tensor, Error> {
    assert_eq!(logits.shape(), target_probs.shape());
    assert_eq!(logits.dtype(), target_probs.dtype());

    let last_dim = *logits.shape.last().unwrap();
    logits
        .log_softmax_along(-1)?
        .sub(target_probs.clone().ln()?)?
        .mul(target_probs)?
        .mean()?
        .negate()?
        .mul_scalar(last_dim)
}

pub fn binary_xent_with_logits(logits: Tensor, target_probs: Tensor) -> Result<Tensor, Error> {
    assert_eq!(logits.dtype(), target_probs.dtype());
    assert_eq!(logits.shape(), target_probs.shape());

    let dtype = logits.dtype();
    let a = logits.clone().max_scalar(dtype.zero())?; // NOTE: fused
    let b = logits.clone().mul(target_probs)?; // NOTE: eager
    let c = logits
        .abs()?
        .negate()?
        .exp()?
        .add_scalar(dtype.one())?
        .ln()?; // NOTE: fused
    a.sub(b)?.add(c)?.mean()
}

pub fn xent_with_logits(logits: Tensor, target_probs: Tensor) -> Result<Tensor, Error> {
    assert_eq!(logits.shape(), target_probs.shape());
    assert_eq!(logits.dtype(), target_probs.dtype());

    let last_dim = *logits.shape.last().unwrap();
    logits
        .log_softmax_along(-1)?
        .mul(target_probs)?
        .mean()?
        .negate()?
        .mul_scalar(last_dim)
}
