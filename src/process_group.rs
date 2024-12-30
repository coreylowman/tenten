struct NcclProcessGroup {
    pub rank: usize,
    pub world_size: usize,
    comm: cudarc::nccl::Comm,
    device_byte: cudarc::driver::CudaSlice<u8>,
}

impl NcclProcessGroup {
    pub fn from_env_vars() -> Result<Self, std::env::VarError> {
        let meeting_addr: SocketAddr = std::env::var("MEETING_ADDR")?.parse().unwrap();
        let rank: usize = std::env::var("RANK")?.parse().unwrap();
        let world_size: usize = std::env::var("WORLD_SIZE")?.parse().unwrap();
        Ok(Self::new(meeting_addr, rank, world_size))
    }

    pub fn new(meeting_addr: SocketAddr, rank: usize, world_size: usize) -> Self {
        use cudarc::driver::CudaDevice;
        use cudarc::nccl::{Comm, Id};

        let local_rank = rank % CudaDevice::count().unwrap() as usize;
        let device = CudaDevice::new(local_rank).unwrap();

        let id = if rank == 0 {
            let listener = std::net::TcpListener::bind(&meeting_addr).unwrap();
            let id = Id::new().unwrap();
            let data = id.internal().map(|i| i as u8);
            for _ in 0..world_size - 1 {
                let (mut stream, _) = listener.accept().unwrap();
                stream.write_all(&data).unwrap();
            }
            id
        } else {
            let mut buf: [u8; 128] = [0; 128];
            let mut stream =
                std::net::TcpStream::connect_timeout(&meeting_addr, Duration::from_secs(60))
                    .unwrap();
            stream.read_exact(&mut buf).unwrap();
            let internal = buf.map(|u| u as i8);
            Id::uninit(internal)
        };

        cudarc::nccl::group_start().unwrap();

        let comm = Comm::from_rank(device, rank, world_size, id).unwrap();

        cudarc::nccl::group_end().unwrap();

        let device_byte = comm.device().alloc_zeros(1).unwrap();

        Self {
            rank,
            world_size,
            comm,
            device_byte,
        }
    }

    pub fn barrier(&self) {
        todo!("in place all gather")
        // self.comm.all_gather(sendbuff, recvbuff)
    }
}
