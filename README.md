# size reductions
- ideally we can collapse axes in place. how does this work for striding?
- i *think* we just keep the strides like they were? and just change the shape of the axis to 1. this will keep the same striding pattern so we can skip
- but how will this work if we broadcast it right after? then the shape is back to the same and we would then stride for each element instead of having 0 stride

shape: Vec<usize>
strides: Vec<usize>
storage: DevicePtr
allocation: Allocation {
    num_bytes: usize
    byte_stride: usize
    physical_shape: Vec<usize>
}


# activation checkpointing

rerun forward again in backwards?
