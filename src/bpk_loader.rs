//! Zero-dependency BurnPack (`NRUB`) reader for loading `ada_vq` (`vq_best.bpk`).

use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::sys::ErrorCode;

#[derive(Clone, Debug)]
pub struct BpkTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub data_start: usize,
    pub data_end: usize,
}

#[derive(Clone, Debug)]
pub struct BpkArchive {
    pub tensors: Vec<BpkTensorInfo>,
    pub payload: Vec<u8>,
}

impl BpkArchive {
    pub fn find(&self, name: &str) -> Option<&BpkTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn read_f32_as_f64(&self, name: &str) -> Result<(Vec<usize>, Vec<f64>), ErrorCode> {
        let info = self.find(name).ok_or(ErrorCode::ShapeMismatch)?;
        let bytes = &self.payload[info.data_start..info.data_end];
        if !bytes.len().is_multiple_of(4) {
            return Err(ErrorCode::ShapeMismatch);
        }
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for c in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64);
        }
        Ok((info.shape.clone(), out))
    }

    pub fn total_f32_params(&self) -> usize {
        self.tensors
            .iter()
            .map(|t| (t.data_end - t.data_start) / 4)
            .sum()
    }
}

struct CborCursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> CborCursor<'a> {
    fn read_u8(&mut self) -> Result<u8, ErrorCode> {
        if self.pos >= self.buf.len() {
            return Err(ErrorCode::ShapeMismatch);
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_arg(&mut self, add: u8) -> Result<u64, ErrorCode> {
        match add {
            0..=23 => Ok(add as u64),
            24 => Ok(self.read_u8()? as u64),
            25 => {
                if self.pos + 2 > self.buf.len() {
                    return Err(ErrorCode::ShapeMismatch);
                }
                let v = u16::from_be_bytes([self.buf[self.pos], self.buf[self.pos + 1]]) as u64;
                self.pos += 2;
                Ok(v)
            }
            26 => {
                if self.pos + 4 > self.buf.len() {
                    return Err(ErrorCode::ShapeMismatch);
                }
                let v =
                    u32::from_be_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap()) as u64;
                self.pos += 4;
                Ok(v)
            }
            27 => {
                if self.pos + 8 > self.buf.len() {
                    return Err(ErrorCode::ShapeMismatch);
                }
                let v = u64::from_be_bytes(self.buf[self.pos..self.pos + 8].try_into().unwrap());
                self.pos += 8;
                Ok(v)
            }
            _ => Err(ErrorCode::InvalidInput),
        }
    }

    fn read_text(&mut self) -> Result<String, ErrorCode> {
        let head = self.read_u8()?;
        let major = head >> 5;
        let add = head & 0x1F;
        if major != 3 {
            return Err(ErrorCode::InvalidInput);
        }
        let len = self.read_arg(add)? as usize;
        if self.pos + len > self.buf.len() {
            return Err(ErrorCode::ShapeMismatch);
        }
        let s = std::str::from_utf8(&self.buf[self.pos..self.pos + len])
            .map_err(|_| ErrorCode::InvalidInput)?
            .to_string();
        self.pos += len;
        Ok(s)
    }

    fn skip_value(&mut self) -> Result<(), ErrorCode> {
        let head = self.read_u8()?;
        let major = head >> 5;
        let add = head & 0x1F;
        match major {
            0 | 1 => {
                let _ = self.read_arg(add)?;
            }
            2 | 3 => {
                let len = self.read_arg(add)? as usize;
                if self.pos + len > self.buf.len() {
                    return Err(ErrorCode::ShapeMismatch);
                }
                self.pos += len;
            }
            4 => {
                let count = self.read_arg(add)? as usize;
                for _ in 0..count {
                    self.skip_value()?;
                }
            }
            5 => {
                let count = self.read_arg(add)? as usize;
                for _ in 0..(2 * count) {
                    self.skip_value()?;
                }
            }
            6 => {
                let _ = self.read_arg(add)?;
                self.skip_value()?;
            }
            7 => {
                if add == 24 {
                    let _ = self.read_u8()?;
                } else if add == 25 {
                    self.pos += 2;
                } else if add == 26 {
                    self.pos += 4;
                } else if add == 27 {
                    self.pos += 8;
                }
            }
            _ => return Err(ErrorCode::InvalidInput),
        }
        Ok(())
    }

    fn read_uint(&mut self) -> Result<usize, ErrorCode> {
        let head = self.read_u8()?;
        if (head >> 5) != 0 {
            return Err(ErrorCode::InvalidInput);
        }
        Ok(self.read_arg(head & 0x1F)? as usize)
    }
}

/// Load a BurnPack (`.bpk`) file into a `BpkArchive`.
pub fn load_bpk(path: &Path) -> Result<BpkArchive, ErrorCode> {
    let mut f = File::open(path).map_err(|_| ErrorCode::InvalidInput)?;
    let mut raw = Vec::new();
    f.read_to_end(&mut raw)
        .map_err(|_| ErrorCode::InvalidInput)?;

    if raw.len() < 10 || &raw[0..4] != b"NRUB" {
        return Err(ErrorCode::InvalidInput);
    }
    let hlen = u32::from_le_bytes(raw[6..10].try_into().unwrap()) as usize;
    let data_base = (10 + hlen).next_multiple_of(256);
    if data_base > raw.len() {
        return Err(ErrorCode::ShapeMismatch);
    }
    let hdr = &raw[10..10 + hlen];
    let payload = raw[data_base..].to_vec();

    let mut cur = CborCursor { buf: hdr, pos: 0 };
    let root_head = cur.read_u8()?;
    if (root_head >> 5) != 5 {
        return Err(ErrorCode::InvalidInput);
    }
    let root_len = cur.read_arg(root_head & 0x1F)? as usize;

    let mut tensors = Vec::new();
    for _ in 0..root_len {
        let k = cur.read_text()?;
        if k == "tensors" {
            let t_head = cur.read_u8()?;
            if (t_head >> 5) != 5 {
                return Err(ErrorCode::InvalidInput);
            }
            let t_len = cur.read_arg(t_head & 0x1F)? as usize;
            for _ in 0..t_len {
                let name = cur.read_text()?;
                let m_head = cur.read_u8()?;
                if (m_head >> 5) != 5 {
                    return Err(ErrorCode::InvalidInput);
                }
                let m_len = cur.read_arg(m_head & 0x1F)? as usize;
                let mut shape = Vec::new();
                let mut data_start = 0usize;
                let mut data_end = 0usize;
                for _ in 0..m_len {
                    let field = cur.read_text()?;
                    match field.as_str() {
                        "shape" => {
                            let a_head = cur.read_u8()?;
                            let a_len = cur.read_arg(a_head & 0x1F)? as usize;
                            for _ in 0..a_len {
                                shape.push(cur.read_uint()?);
                            }
                        }
                        "data_offsets" => {
                            let a_head = cur.read_u8()?;
                            let a_len = cur.read_arg(a_head & 0x1F)? as usize;
                            if a_len != 2 {
                                return Err(ErrorCode::ShapeMismatch);
                            }
                            data_start = cur.read_uint()?;
                            data_end = cur.read_uint()?;
                        }
                        _ => cur.skip_value()?,
                    }
                }
                if data_end > payload.len() || data_end < data_start {
                    return Err(ErrorCode::ShapeMismatch);
                }
                tensors.push(BpkTensorInfo {
                    name,
                    shape,
                    data_start,
                    data_end,
                });
            }
        } else {
            cur.skip_value()?;
        }
    }

    Ok(BpkArchive { tensors, payload })
}
