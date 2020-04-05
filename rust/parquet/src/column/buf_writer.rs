// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Contains column writer API.

use std::{cmp, collections::VecDeque, rc::Rc, io::{Write, Seek, Cursor, SeekFrom}};

use crate::basic::{Compression, Encoding, PageType, Type};
use crate::column::page::{CompressedPage, Page, PageWriteSpec, PageWriter};
use crate::compression::{create_codec, Codec};
use crate::data_type::*;
use crate::encodings::{
    encoding::{get_encoder, DictEncoder, Encoder},
    levels::{max_buffer_size, LevelEncoder},
};
use crate::errors::{ParquetError, Result};
use crate::file::{
    metadata::ColumnChunkMetaData,
    properties::{WriterProperties, WriterPropertiesPtr, WriterVersion},
};
use crate::schema::types::ColumnDescPtr;
use crate::util::memory::{ByteBufferPtr, MemTracker};

/// Column writer for a Parquet type.
pub enum ColumnWriter {
    BoolColumnWriter(ColumnWriterImpl<BoolType>),
    Int32ColumnWriter(ColumnWriterImpl<Int32Type>),
    Int64ColumnWriter(ColumnWriterImpl<Int64Type>),
    Int96ColumnWriter(ColumnWriterImpl<Int96Type>),
    FloatColumnWriter(ColumnWriterImpl<FloatType>),
    DoubleColumnWriter(ColumnWriterImpl<DoubleType>),
    ByteArrayColumnWriter(ColumnWriterImpl<ByteArrayType>),
    FixedLenByteArrayColumnWriter(ColumnWriterImpl<FixedLenByteArrayType>),
}

/// Gets a specific column writer corresponding to column descriptor `descr`.
pub fn get_column_writer(
    descr: ColumnDescPtr,
    props: WriterPropertiesPtr,
) -> ColumnWriter {
    match descr.physical_type() {
        Type::BOOLEAN => ColumnWriter::BoolColumnWriter(ColumnWriterImpl::new(
            descr,
            props,
        )),
        Type::INT32 => ColumnWriter::Int32ColumnWriter(ColumnWriterImpl::new(
            descr,
            props,
        )),
        Type::INT64 => ColumnWriter::Int64ColumnWriter(ColumnWriterImpl::new(
            descr,
            props,
        )),
        Type::INT96 => ColumnWriter::Int96ColumnWriter(ColumnWriterImpl::new(
            descr,
            props,
        )),
        Type::FLOAT => ColumnWriter::FloatColumnWriter(ColumnWriterImpl::new(
            descr,
            props,
        )),
        Type::DOUBLE => ColumnWriter::DoubleColumnWriter(ColumnWriterImpl::new(
            descr,
            props,
        )),
        Type::BYTE_ARRAY => ColumnWriter::ByteArrayColumnWriter(ColumnWriterImpl::new(
            descr,
            props,
        )),
        Type::FIXED_LEN_BYTE_ARRAY => ColumnWriter::FixedLenByteArrayColumnWriter(
            ColumnWriterImpl::new(descr, props),
        ),
    }
}

// /// Gets a typed column writer for the specific type `T`, by "up-casting" `col_writer` of
// /// non-generic type to a generic column writer type `ColumnWriterImpl`.
// ///
// /// Panics if actual enum value for `col_writer` does not match the type `T`.
// pub fn get_typed_column_writer<T: DataType>(
//     col_writer: ColumnWriter,
// ) -> ColumnWriterImpl<T> {
//     T::get_buf_column_writer(col_writer).unwrap_or_else(|| {
//         panic!(
//             "Failed to convert column writer into a typed column writer for `{}` type",
//             T::get_physical_type()
//         )
//     })
// }

// /// Similar to `get_typed_column_writer` but returns a reference.
// pub fn get_typed_column_writer_ref<T: DataType>(
//     col_writer: &ColumnWriter,
// ) -> &ColumnWriterImpl<T> {
//     T::get_column_writer_ref(col_writer).unwrap_or_else(|| {
//         panic!(
//             "Failed to convert column writer into a typed column writer for `{}` type",
//             T::get_physical_type()
//         )
//     })
// }

// /// Similar to `get_typed_column_writer` but returns a reference.
// pub fn get_typed_column_writer_mut<T: DataType>(
//     col_writer: &mut ColumnWriter,
// ) -> &mut ColumnWriterImpl<T> {
//     T::get_column_writer_mut(col_writer).unwrap_or_else(|| {
//         panic!(
//             "Failed to convert column writer into a typed column writer for `{}` type",
//             T::get_physical_type()
//         )
//     })
// }

/// Typed column writer for a primitive column.
pub struct ColumnWriterImpl<T: DataType> {
    // Column writer properties
    descr: ColumnDescPtr,
    props: WriterPropertiesPtr,
    buf: Box<Vec<u8>>,
    page_writer: Box<PageWriter>,
    has_dictionary: bool,
    dict_encoder: Option<DictEncoder<T>>,
    encoder: Box<Encoder<T>>,
    codec: Compression,
    compressor: Option<Box<Codec>>,
    // Metrics per page
    num_buffered_values: u32,
    num_buffered_encoded_values: u32,
    num_buffered_rows: u32,
    // Metrics per column writer
    total_bytes_written: u64,
    total_rows_written: u64,
    total_uncompressed_size: u64,
    total_compressed_size: u64,
    total_num_values: u64,
    dictionary_page_offset: Option<u64>,
    data_page_offset: Option<u64>,
    // Reused buffers
    def_levels_sink: Vec<i16>,
    rep_levels_sink: Vec<i16>,
    data_pages: VecDeque<CompressedPage>,
}

impl<T: DataType> ColumnWriterImpl<T> {
    pub fn new(
        descr: ColumnDescPtr,
        props: WriterPropertiesPtr,
    ) -> Self {
        let buf_ptr = Box::into_raw(Box::new(Vec::<u8>::with_capacity(16 * 1024 * 1024)));
        let cursor = Cursor::new(unsafe { &mut *buf_ptr });
        let page_writer = Box::new(crate::file::writer::SerializedPageWriter::new(cursor));
        let buf = unsafe { Box::from_raw(buf_ptr) };

        let codec = props.compression(descr.path());
        let compressor = create_codec(codec).unwrap();

        // Optionally set dictionary encoder.
        let dict_encoder = if props.dictionary_enabled(descr.path())
            && Self::has_dictionary_support(&props)
        {
            Some(DictEncoder::new(descr.clone(), Rc::new(MemTracker::new())))
        } else {
            None
        };

        // Whether or not this column writer has a dictionary encoding.
        let has_dictionary = dict_encoder.is_some();

        // Set either main encoder or fallback encoder.
        let fallback_encoder = get_encoder(
            descr.clone(),
            props
                .encoding(descr.path())
                .unwrap_or(Self::fallback_encoding(&props)),
            Rc::new(MemTracker::new()),
        )
        .unwrap();

        Self {
            descr,
            props,
            buf: buf,
            page_writer,
            has_dictionary,
            dict_encoder,
            encoder: fallback_encoder,
            codec,
            compressor,
            num_buffered_values: 0,
            num_buffered_encoded_values: 0,
            num_buffered_rows: 0,
            total_bytes_written: 0,
            total_rows_written: 0,
            total_uncompressed_size: 0,
            total_compressed_size: 0,
            total_num_values: 0,
            dictionary_page_offset: None,
            data_page_offset: None,
            def_levels_sink: vec![],
            rep_levels_sink: vec![],
            data_pages: VecDeque::new(),
        }
    }

    /// Writes batch of values, definition levels and repetition levels.
    /// Returns number of values processed (written).
    ///
    /// If definition and repetition levels are provided, we write fully those levels and
    /// select how many values to write (this number will be returned), since number of
    /// actual written values may be smaller than provided values.
    ///
    /// If only values are provided, then all values are written and the length of
    /// of the values buffer is returned.
    ///
    /// Definition and/or repetition levels can be omitted, if values are
    /// non-nullable and/or non-repeated.
    pub fn write_batch(
        &mut self,
        values: &[T::T],
        def_levels: Option<&[i16]>,
        rep_levels: Option<&[i16]>,
    ) -> Result<usize> {
        // We check for DataPage limits only after we have inserted the values. If a user
        // writes a large number of values, the DataPage size can be well above the limit.
        //
        // The purpose of this chunking is to bound this. Even if a user writes large
        // number of values, the chunking will ensure that we add data page at a
        // reasonable pagesize limit.

        // TODO: find out why we don't account for size of levels when we estimate page
        // size.

        // Find out the minimal length to prevent index out of bound errors.
        let mut min_len = values.len();
        if let Some(levels) = def_levels {
            min_len = cmp::min(min_len, levels.len());
        }
        if let Some(levels) = rep_levels {
            min_len = cmp::min(min_len, levels.len());
        }

        // Find out number of batches to process.
        let write_batch_size = self.props.write_batch_size();
        let num_batches = min_len / write_batch_size;

        let mut values_offset = 0;
        let mut levels_offset = 0;

        for _ in 0..num_batches {
            values_offset += self.write_mini_batch(
                &values[values_offset..values_offset + write_batch_size],
                def_levels.map(|lv| &lv[levels_offset..levels_offset + write_batch_size]),
                rep_levels.map(|lv| &lv[levels_offset..levels_offset + write_batch_size]),
            )?;
            levels_offset += write_batch_size;
        }

        values_offset += self.write_mini_batch(
            &values[values_offset..],
            def_levels.map(|lv| &lv[levels_offset..]),
            rep_levels.map(|lv| &lv[levels_offset..]),
        )?;

        // Return total number of values processed.
        Ok(values_offset)
    }

    /// Returns total number of bytes written by this column writer so far.
    /// This value is also returned when column writer is closed.
    pub fn get_total_bytes_written(&self) -> u64 {
        self.total_bytes_written
    }

    /// Returns total number of rows written by this column writer so far.
    /// This value is also returned when column writer is closed.
    pub fn get_total_rows_written(&self) -> u64 {
        self.total_rows_written
    }

    /// Finalises writes and closes the column writer.
    /// Returns total bytes written, total rows written and column chunk metadata.
    pub fn close<S>(mut self, sink: &mut S) -> Result<(u64, u64, ColumnChunkMetaData)> 
    where S: Sized + Write + Seek
    {
        if self.dict_encoder.is_some() {
            self.write_dictionary_page()?;
        }
        self.flush_data_pages()?;
        let sink_pos = sink.seek(SeekFrom::Current(0)).unwrap();
        let metadata = self.write_column_metadata(sink_pos)?;
        self.dict_encoder = None;
        self.page_writer.close()?;

        std::io::copy(&mut &self.buf[..], sink).unwrap();

        Ok((self.total_bytes_written, self.total_rows_written, metadata))
    }

    /// Writes mini batch of values, definition and repetition levels.
    /// This allows fine-grained processing of values and maintaining a reasonable
    /// page size.
    fn write_mini_batch(
        &mut self,
        values: &[T::T],
        def_levels: Option<&[i16]>,
        rep_levels: Option<&[i16]>,
    ) -> Result<usize> {
        let num_values;
        let mut values_to_write = 0;

        // Check if number of definition levels is the same as number of repetition
        // levels.
        if def_levels.is_some() && rep_levels.is_some() {
            let def = def_levels.unwrap();
            let rep = rep_levels.unwrap();
            if def.len() != rep.len() {
                return Err(general_err!(
                    "Inconsistent length of definition and repetition levels: {} != {}",
                    def.len(),
                    rep.len()
                ));
            }
        }

        // Process definition levels and determine how many values to write.
        if self.descr.max_def_level() > 0 {
            if def_levels.is_none() {
                return Err(general_err!(
                    "Definition levels are required, because max definition level = {}",
                    self.descr.max_def_level()
                ));
            }

            let levels = def_levels.unwrap();
            num_values = levels.len();
            for &level in levels {
                values_to_write += (level == self.descr.max_def_level()) as usize;
            }

            self.write_definition_levels(levels);
        } else {
            values_to_write = values.len();
            num_values = values_to_write;
        }

        // Process repetition levels and determine how many rows we are about to process.
        if self.descr.max_rep_level() > 0 {
            // A row could contain more than one value.
            if rep_levels.is_none() {
                return Err(general_err!(
                    "Repetition levels are required, because max repetition level = {}",
                    self.descr.max_rep_level()
                ));
            }

            // Count the occasions where we start a new row
            let levels = rep_levels.unwrap();
            for &level in levels {
                self.num_buffered_rows += (level == 0) as u32
            }

            self.write_repetition_levels(levels);
        } else {
            // Each value is exactly one row.
            // Equals to the number of values, we count nulls as well.
            self.num_buffered_rows += num_values as u32;
        }

        // Check that we have enough values to write.
        if values.len() < values_to_write {
            return Err(general_err!(
                "Expected to write {} values, but have only {}",
                values_to_write,
                values.len()
            ));
        }

        // TODO: update page statistics

        self.write_values(&values[0..values_to_write])?;

        self.num_buffered_values += num_values as u32;
        self.num_buffered_encoded_values += values_to_write as u32;

        if self.should_add_data_page() {
            self.add_data_page()?;
        }

        if self.should_dict_fallback() {
            self.dict_fallback()?;
        }

        Ok(values_to_write)
    }

    #[inline]
    fn write_definition_levels(&mut self, def_levels: &[i16]) {
        self.def_levels_sink.extend_from_slice(def_levels);
    }

    #[inline]
    fn write_repetition_levels(&mut self, rep_levels: &[i16]) {
        self.rep_levels_sink.extend_from_slice(rep_levels);
    }

    #[inline]
    fn write_values(&mut self, values: &[T::T]) -> Result<()> {
        match self.dict_encoder {
            Some(ref mut encoder) => encoder.put(values),
            None => self.encoder.put(values),
        }
    }

    /// Returns true if we need to fall back to non-dictionary encoding.
    ///
    /// We can only fall back if dictionary encoder is set and we have exceeded dictionary
    /// size.
    #[inline]
    fn should_dict_fallback(&self) -> bool {
        match self.dict_encoder {
            Some(ref encoder) => {
                encoder.dict_encoded_size() >= self.props.dictionary_pagesize_limit()
            }
            None => false,
        }
    }

    /// Returns true if there is enough data for a data page, false otherwise.
    #[inline]
    fn should_add_data_page(&self) -> bool {
        match self.dict_encoder {
            Some(ref encoder) => {
                encoder.estimated_data_encoded_size() >= self.props.data_pagesize_limit()
            }
            None => {
                self.encoder.estimated_data_encoded_size()
                    >= self.props.data_pagesize_limit()
            }
        }
    }

    /// Performs dictionary fallback.
    /// Prepares and writes dictionary and all data pages into page writer.
    fn dict_fallback(&mut self) -> Result<()> {
        // At this point we know that we need to fall back.
        self.write_dictionary_page()?;
        self.flush_data_pages()?;
        self.dict_encoder = None;
        Ok(())
    }

    /// Adds data page.
    /// Data page is either buffered in case of dictionary encoding or written directly.
    fn add_data_page(&mut self) -> Result<()> {
        // Extract encoded values
        let value_bytes = match self.dict_encoder {
            Some(ref mut encoder) => encoder.write_indices()?,
            None => self.encoder.flush_buffer()?,
        };

        // Select encoding based on current encoder and writer version (v1 or v2).
        let encoding = if self.dict_encoder.is_some() {
            self.props.dictionary_data_page_encoding()
        } else {
            self.encoder.encoding()
        };

        let max_def_level = self.descr.max_def_level();
        let max_rep_level = self.descr.max_rep_level();

        let compressed_page = match self.props.writer_version() {
            WriterVersion::PARQUET_1_0 => {
                let mut buffer = vec![];

                if max_rep_level > 0 {
                    buffer.extend_from_slice(
                        &self.encode_levels_v1(
                            Encoding::RLE,
                            &self.rep_levels_sink[..],
                            max_rep_level,
                        )?[..],
                    );
                }

                if max_def_level > 0 {
                    buffer.extend_from_slice(
                        &self.encode_levels_v1(
                            Encoding::RLE,
                            &self.def_levels_sink[..],
                            max_def_level,
                        )?[..],
                    );
                }

                buffer.extend_from_slice(value_bytes.data());
                let uncompressed_size = buffer.len();

                if let Some(ref mut cmpr) = self.compressor {
                    let mut compressed_buf = Vec::with_capacity(value_bytes.data().len());
                    cmpr.compress(&buffer[..], &mut compressed_buf)?;
                    buffer = compressed_buf;
                }

                let data_page = Page::DataPage {
                    buf: ByteBufferPtr::new(buffer),
                    num_values: self.num_buffered_values,
                    encoding,
                    def_level_encoding: Encoding::RLE,
                    rep_level_encoding: Encoding::RLE,
                    // TODO: process statistics
                    statistics: None,
                };

                CompressedPage::new(data_page, uncompressed_size)
            }
            WriterVersion::PARQUET_2_0 => {
                let mut rep_levels_byte_len = 0;
                let mut def_levels_byte_len = 0;
                let mut buffer = vec![];

                if max_rep_level > 0 {
                    let levels =
                        self.encode_levels_v2(&self.rep_levels_sink[..], max_rep_level)?;
                    rep_levels_byte_len = levels.len();
                    buffer.extend_from_slice(&levels[..]);
                }

                if max_def_level > 0 {
                    let levels =
                        self.encode_levels_v2(&self.def_levels_sink[..], max_def_level)?;
                    def_levels_byte_len = levels.len();
                    buffer.extend_from_slice(&levels[..]);
                }

                let uncompressed_size =
                    rep_levels_byte_len + def_levels_byte_len + value_bytes.len();

                // Data Page v2 compresses values only.
                match self.compressor {
                    Some(ref mut cmpr) => {
                        let mut compressed_buf =
                            Vec::with_capacity(value_bytes.data().len());
                        cmpr.compress(value_bytes.data(), &mut compressed_buf)?;
                        buffer.extend_from_slice(&compressed_buf[..]);
                    }
                    None => {
                        buffer.extend_from_slice(value_bytes.data());
                    }
                }

                let data_page = Page::DataPageV2 {
                    buf: ByteBufferPtr::new(buffer),
                    num_values: self.num_buffered_values,
                    encoding,
                    num_nulls: self.num_buffered_values
                        - self.num_buffered_encoded_values,
                    num_rows: self.num_buffered_rows,
                    def_levels_byte_len: def_levels_byte_len as u32,
                    rep_levels_byte_len: rep_levels_byte_len as u32,
                    is_compressed: self.compressor.is_some(),
                    // TODO: process statistics
                    statistics: None,
                };

                CompressedPage::new(data_page, uncompressed_size)
            }
        };

        // Check if we need to buffer data page or flush it to the sink directly.
        if self.dict_encoder.is_some() {
            self.data_pages.push_back(compressed_page);
        } else {
            self.write_data_page(compressed_page)?;
        }

        // Update total number of rows.
        self.total_rows_written += self.num_buffered_rows as u64;

        // Reset state.
        self.rep_levels_sink.clear();
        self.def_levels_sink.clear();
        self.num_buffered_values = 0;
        self.num_buffered_encoded_values = 0;
        self.num_buffered_rows = 0;

        Ok(())
    }

    /// Finalises any outstanding data pages and flushes buffered data pages from
    /// dictionary encoding into underlying sink.
    #[inline]
    fn flush_data_pages(&mut self) -> Result<()> {
        // Write all outstanding data to a new page.
        if self.num_buffered_values > 0 {
            self.add_data_page()?;
        }

        while let Some(page) = self.data_pages.pop_front() {
            self.write_data_page(page)?;
        }

        Ok(())
    }

    /// Assembles and writes column chunk metadata.
    fn write_column_metadata(&mut self, sink_pos: u64) -> Result<ColumnChunkMetaData> 
    {
        // let sink_pos = sink.seek(SeekFrom::Current(0)).unwrap();
        let total_compressed_size = self.total_compressed_size as i64;
        let total_uncompressed_size = self.total_uncompressed_size as i64;
        let num_values = self.total_num_values as i64;
        let dict_page_offset = self.dictionary_page_offset.map(|v| (v + sink_pos) as i64);
        // If data page offset is not set, then no pages have been written
        let data_page_offset = self.data_page_offset.map(|v| (v + sink_pos) as i64).unwrap_or(0i64);

        let file_offset;
        let mut encodings = Vec::new();

        if self.has_dictionary {
            assert!(dict_page_offset.is_some(), "Dictionary offset is not set");
            file_offset = dict_page_offset.unwrap() + total_compressed_size;
            // NOTE: This should be in sync with writing dictionary pages.
            encodings.push(self.props.dictionary_page_encoding());
            encodings.push(self.props.dictionary_data_page_encoding());
            // Fallback to alternative encoding, add it to the list.
            if self.dict_encoder.is_none() {
                encodings.push(self.encoder.encoding());
            }
        } else {
            file_offset = data_page_offset + total_compressed_size;
            encodings.push(self.encoder.encoding());
        }
        // We use only RLE level encoding for data page v1 and data page v2.
        encodings.push(Encoding::RLE);

        let metadata = ColumnChunkMetaData::builder(self.descr.clone())
            .set_compression(self.codec)
            .set_encodings(encodings)
            .set_file_offset(file_offset)
            .set_total_compressed_size(total_compressed_size)
            .set_total_uncompressed_size(total_uncompressed_size)
            .set_num_values(num_values)
            .set_data_page_offset(data_page_offset)
            .set_dictionary_page_offset(dict_page_offset)
            .build()?;

        self.page_writer.write_metadata(&metadata)?;

        Ok(metadata)
    }

    /// Encodes definition or repetition levels for Data Page v1.
    #[inline]
    fn encode_levels_v1(
        &self,
        encoding: Encoding,
        levels: &[i16],
        max_level: i16,
    ) -> Result<Vec<u8>> {
        let size = max_buffer_size(encoding, max_level, levels.len());
        let mut encoder = LevelEncoder::v1(encoding, max_level, vec![0; size]);
        encoder.put(&levels)?;
        encoder.consume()
    }

    /// Encodes definition or repetition levels for Data Page v2.
    /// Encoding is always RLE.
    #[inline]
    fn encode_levels_v2(&self, levels: &[i16], max_level: i16) -> Result<Vec<u8>> {
        let size = max_buffer_size(Encoding::RLE, max_level, levels.len());
        let mut encoder = LevelEncoder::v2(max_level, vec![0; size]);
        encoder.put(&levels)?;
        encoder.consume()
    }

    /// Writes compressed data page into underlying sink and updates global metrics.
    #[inline]
    fn write_data_page(&mut self, page: CompressedPage) -> Result<()> {
        let page_spec = self.page_writer.write_page(page)?;
        self.update_metrics_for_page(page_spec);
        Ok(())
    }

    /// Writes dictionary page into underlying sink.
    #[inline]
    fn write_dictionary_page(&mut self) -> Result<()> {
        if self.dict_encoder.is_none() {
            return Err(general_err!("Dictionary encoder is not set"));
        }

        let compressed_page = {
            let encoder = self.dict_encoder.as_ref().unwrap();
            let is_sorted = encoder.is_sorted();
            let num_values = encoder.num_entries();
            let mut values_buf = encoder.write_dict()?;
            let uncompressed_size = values_buf.len();

            if let Some(ref mut cmpr) = self.compressor {
                let mut output_buf = Vec::with_capacity(uncompressed_size);
                cmpr.compress(values_buf.data(), &mut output_buf)?;
                values_buf = ByteBufferPtr::new(output_buf);
            }

            let dict_page = Page::DictionaryPage {
                buf: values_buf,
                num_values: num_values as u32,
                encoding: self.props.dictionary_page_encoding(),
                is_sorted,
            };
            CompressedPage::new(dict_page, uncompressed_size)
        };

        let page_spec = self.page_writer.write_page(compressed_page)?;
        self.update_metrics_for_page(page_spec);
        Ok(())
    }

    /// Updates column writer metrics with each page metadata.
    #[inline]
    fn update_metrics_for_page(&mut self, page_spec: PageWriteSpec) {
        self.total_uncompressed_size += page_spec.uncompressed_size as u64;
        self.total_compressed_size += page_spec.compressed_size as u64;
        self.total_num_values += page_spec.num_values as u64;
        self.total_bytes_written += page_spec.bytes_written;

        match page_spec.page_type {
            PageType::DATA_PAGE | PageType::DATA_PAGE_V2 => {
                if self.data_page_offset.is_none() {
                    self.data_page_offset = Some(page_spec.offset);
                }
            }
            PageType::DICTIONARY_PAGE => {
                assert!(
                    self.dictionary_page_offset.is_none(),
                    "Dictionary offset is already set"
                );
                self.dictionary_page_offset = Some(page_spec.offset);
            }
            _ => {}
        }
    }

    /// Returns reference to the underlying page writer.
    /// This method is intended to use in tests only.
    fn get_page_writer_ref(&self) -> &Box<PageWriter> {
        &self.page_writer
    }
}

// ----------------------------------------------------------------------
// Encoding support for column writer.
// This mirrors parquet-mr default encodings for writes. See:
// https://github.com/apache/parquet-mr/blob/master/parquet-column/src/main/java/org/apache/parquet/column/values/factory/DefaultV1ValuesWriterFactory.java
// https://github.com/apache/parquet-mr/blob/master/parquet-column/src/main/java/org/apache/parquet/column/values/factory/DefaultV2ValuesWriterFactory.java

/// Trait to define default encoding for types, including whether or not the type
/// supports dictionary encoding.
trait EncodingWriteSupport {
    /// Returns encoding for a column when no other encoding is provided in writer
    /// properties.
    fn fallback_encoding(props: &WriterProperties) -> Encoding;

    /// Returns true if dictionary is supported for column writer, false otherwise.
    fn has_dictionary_support(props: &WriterProperties) -> bool;
}

// Basic implementation, always falls back to PLAIN and supports dictionary.
impl<T: DataType> EncodingWriteSupport for ColumnWriterImpl<T> {
    default fn fallback_encoding(_props: &WriterProperties) -> Encoding {
        Encoding::PLAIN
    }

    default fn has_dictionary_support(_props: &WriterProperties) -> bool {
        true
    }
}

impl EncodingWriteSupport for ColumnWriterImpl<BoolType> {
    fn fallback_encoding(props: &WriterProperties) -> Encoding {
        match props.writer_version() {
            WriterVersion::PARQUET_1_0 => Encoding::PLAIN,
            WriterVersion::PARQUET_2_0 => Encoding::RLE,
        }
    }

    // Boolean column does not support dictionary encoding and should fall back to
    // whatever fallback encoding is defined.
    fn has_dictionary_support(_props: &WriterProperties) -> bool {
        false
    }
}

impl EncodingWriteSupport for ColumnWriterImpl<Int32Type> {
    fn fallback_encoding(props: &WriterProperties) -> Encoding {
        match props.writer_version() {
            WriterVersion::PARQUET_1_0 => Encoding::PLAIN,
            WriterVersion::PARQUET_2_0 => Encoding::DELTA_BINARY_PACKED,
        }
    }
}

impl EncodingWriteSupport for ColumnWriterImpl<Int64Type> {
    fn fallback_encoding(props: &WriterProperties) -> Encoding {
        match props.writer_version() {
            WriterVersion::PARQUET_1_0 => Encoding::PLAIN,
            WriterVersion::PARQUET_2_0 => Encoding::DELTA_BINARY_PACKED,
        }
    }
}

impl EncodingWriteSupport for ColumnWriterImpl<ByteArrayType> {
    fn fallback_encoding(props: &WriterProperties) -> Encoding {
        match props.writer_version() {
            WriterVersion::PARQUET_1_0 => Encoding::PLAIN,
            WriterVersion::PARQUET_2_0 => Encoding::DELTA_BYTE_ARRAY,
        }
    }
}

impl EncodingWriteSupport for ColumnWriterImpl<FixedLenByteArrayType> {
    fn fallback_encoding(props: &WriterProperties) -> Encoding {
        match props.writer_version() {
            WriterVersion::PARQUET_1_0 => Encoding::PLAIN,
            WriterVersion::PARQUET_2_0 => Encoding::DELTA_BYTE_ARRAY,
        }
    }

    fn has_dictionary_support(props: &WriterProperties) -> bool {
        match props.writer_version() {
            // Dictionary encoding was not enabled in PARQUET 1.0
            WriterVersion::PARQUET_1_0 => false,
            WriterVersion::PARQUET_2_0 => true,
        }
    }
}
