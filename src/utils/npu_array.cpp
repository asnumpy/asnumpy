/******************************************************************************
 * Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/


 #include <asnumpy/utils/npu_array.hpp>
 #include <asnumpy/memory/MemoryPool.hpp> // 引入内存池
 #include <cstddef>
 #include <iostream>
 
 
 /**
  * @brief Constructor that creates an NPUArray with specified shape and data type.
  */
 NPUArray::NPUArray(const std::vector<int64_t>& shape, py::dtype dtype) {
     this->shape = shape;
     this->dtype = dtype;
     this->aclDtype = GetACLDataType(dtype);
     this->tensorSize = GetShapeSize(shape);
     auto tensorByteSize = this->tensorSize * GetDataTypeSize(this->aclDtype);
     
     // 改用内存池
     this->devicePtr = asnumpy::memory::MemoryPool::instance().malloc(tensorByteSize);
     if(this->devicePtr == nullptr && tensorByteSize > 0) {
         throw std::runtime_error("NPUArray MemoryPool malloc error!");
     }
     
     this->strides.resize(this->shape.size());
     auto currentStride = 1;
     for(int64_t i = this->shape.size() - 1; i >= 0; i--) {
         this->strides[i] = currentStride;
         currentStride *= this->shape[i];
     }
     this->tensorPtr = aclCreateTensor(this->shape.data(), this->shape.size(), GetACLDataType(this->dtype), this->strides.data(), 0, ACL_FORMAT_ND, this->shape.data(), this->shape.size(), this->devicePtr);
 }
 
 
 /**
  * @brief Constructor that creates an NPUArray with specified shape and ACL data type.
  */
 NPUArray::NPUArray(const std::vector<int64_t>& shape, aclDataType acl_type) {
     this->shape = shape;
     this->aclDtype = acl_type;
     this->tensorSize = GetShapeSize(shape);
     auto tensorByteSize = this->tensorSize * GetDataTypeSize(this->aclDtype);
     
     // 直接使用 ACL 类型，不创建 NumPy dtype
     this->dtype = GetPyDtype(acl_type);
     
     // 改用内存池。修复了原代码中未赋值给 this->devicePtr 的问题
     this->devicePtr = asnumpy::memory::MemoryPool::instance().malloc(tensorByteSize);
     if(this->devicePtr == nullptr && tensorByteSize > 0) {
         throw std::runtime_error("NPUArray MemoryPool malloc error!");
     }
     
     this->strides.resize(this->shape.size());
     auto currentStride = 1;
     for(int64_t i = this->shape.size() - 1; i >= 0; i--) {
         this->strides[i] = currentStride;
         currentStride *= this->shape[i];
     }
     this->tensorPtr = aclCreateTensor(this->shape.data(), this->shape.size(), acl_type, this->strides.data(), 0, ACL_FORMAT_ND, this->shape.data(), this->shape.size(), this->devicePtr);
 }
 
 
 /**
  * @brief Copy constructor - deep copy.
  */
 NPUArray::NPUArray(const NPUArray& other) {
     this->shape = other.shape;
     this->dtype = other.dtype;
     this->aclDtype = other.aclDtype;
     this->tensorSize = other.tensorSize;
     this->strides = other.strides;
     auto tensorByteSize = this->tensorSize * GetDataTypeSize(this->aclDtype);
     
     // 劫持分配
     this->devicePtr = asnumpy::memory::MemoryPool::instance().malloc(tensorByteSize);
     if(this->devicePtr == nullptr && tensorByteSize > 0) {
         throw std::runtime_error("NPUArray copy constructor MemoryPool malloc error!");
     }
     
     this->tensorPtr = aclCreateTensor(this->shape.data(), this->shape.size(), this->aclDtype, this->strides.data(), 0, ACL_FORMAT_ND, this->shape.data(), this->shape.size(), this->devicePtr);
     
     void* srcPtr = nullptr;
     auto error = aclGetRawTensorAddr(other.tensorPtr, &srcPtr);
     if(error != ACL_SUCCESS || !srcPtr) throw std::runtime_error(fmt::format("Failed to get source tensor data pointer. error: {}", error));
     
     error = aclrtMemcpy(this->devicePtr, tensorByteSize, srcPtr, tensorByteSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
     if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("Failed to copy tensor data. error: {}", error));
     
     error = aclrtSynchronizeDevice();
     if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("Failed to synchronize after tensor copy. error: {}", error));
 }
 
 
 /**
  * @brief Move constructor.
  */
 NPUArray::NPUArray(NPUArray&& other) noexcept {
     this->tensorPtr = other.tensorPtr;
     this->shape = std::move(other.shape);
     this->dtype = other.dtype;
     this->aclDtype = other.aclDtype;
     this->tensorSize = other.tensorSize;
     this->strides = std::move(other.strides);
     this->devicePtr = other.devicePtr;
     other.tensorPtr = nullptr;
     other.devicePtr = nullptr;
 }
 
 
 /**
  * @brief Copy assignment operator.
  */
 NPUArray& NPUArray::operator=(const NPUArray& other) {
     if(this != &other) {
         if(this->tensorPtr) {
             aclDestroyTensor(this->tensorPtr);
         }
         if(this->devicePtr) {
             asnumpy::memory::MemoryPool::instance().free(this->devicePtr); // 劫持释放
         }
         
         this->shape = other.shape;
         this->dtype = other.dtype;
         this->aclDtype = other.aclDtype;
         this->tensorSize = other.tensorSize;
         this->strides = other.strides;
 
         auto tensorByteSize = this->tensorSize * GetDataTypeSize(this->aclDtype);
         
         // 劫持分配
         this->devicePtr = asnumpy::memory::MemoryPool::instance().malloc(tensorByteSize);
         if(this->devicePtr == nullptr && tensorByteSize > 0) {
             throw std::runtime_error("NPUArray copy assignment MemoryPool malloc error!");
         }
         
         this->tensorPtr = aclCreateTensor(this->shape.data(), this->shape.size(), this->aclDtype, this->strides.data(), 0, ACL_FORMAT_ND, this->shape.data(), this->shape.size(), this->devicePtr);
         
         void* srcPtr = nullptr;
         auto error = aclGetRawTensorAddr(other.tensorPtr, &srcPtr);
         if(error != ACL_SUCCESS || !srcPtr) throw std::runtime_error(fmt::format("Failed to get source tensor data pointer. error: {}", error));
         
         error = aclrtMemcpy(this->devicePtr, tensorByteSize, srcPtr, tensorByteSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
         if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("Failed to copy tensor data. error: {}", error));
         
         error = aclrtSynchronizeDevice();
         if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("Failed to synchronize after tensor copy. error: {}", error));
     }
     return *this;
 }
 
 
 /**
  * @brief Move assignment operator.
  */
 NPUArray& NPUArray::operator=(NPUArray&& other) noexcept {
     if(this != &other) {
         if(this->tensorPtr) {
             aclDestroyTensor(this->tensorPtr);
         }
         if(this->devicePtr) {
             asnumpy::memory::MemoryPool::instance().free(this->devicePtr); // 劫持释放
         }
         
         this->tensorPtr = other.tensorPtr;
         this->shape = std::move(other.shape);
         this->dtype = other.dtype;
         this->aclDtype = other.aclDtype;
         this->tensorSize = other.tensorSize;
         this->strides = std::move(other.strides);
         this->devicePtr = other.devicePtr;
         
         other.tensorPtr = nullptr;
         other.devicePtr = nullptr;
     }
     return *this;
 }
 
 
 /**
  * @brief Destructor that releases resources occupied by NPUArray.
  */
 NPUArray::~NPUArray() {
     if(this->tensorPtr) {
         auto error = aclDestroyTensor(this->tensorPtr);
         this->tensorPtr = nullptr;
     }
     if (this->devicePtr) {
         // 劫持释放：归还给内存池
         asnumpy::memory::MemoryPool::instance().free(this->devicePtr);
         this->devicePtr = nullptr;
     }
 }
 
 
 /**
  * @brief Static method to create NPUArray from NumPy array.
  */
 NPUArray NPUArray::FromNumpy(py::array hostData) {
     py::buffer_info info = hostData.request();
     auto tensorByteSize = info.size * info.itemsize;
     auto result = NPUArray(info.shape, hostData.dtype());
     void* rawDataPtr = nullptr;
     auto error = aclGetRawTensorAddr(result.tensorPtr, &rawDataPtr);
     if (error != ACL_SUCCESS || !rawDataPtr) throw std::runtime_error(fmt::format("Failed to get tensor data pointer. error: {}", error));
     error = aclrtMemcpy(rawDataPtr, tensorByteSize, info.ptr, tensorByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
     if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("Failed to copy numpy data to device. error: {}", error));
     error = aclrtSynchronizeStream(nullptr);
     if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("aclrtSynchronizeStream error: {}", error));
     return result;
 }
 
 
 /**
  * @brief Convert NPUArray to NumPy array.
  */
 py::array NPUArray::ToNumpy() const {
     auto tensorByteSize = this->tensorSize * GetDataTypeSize(this->aclDtype);
     void* rawDataPtr = nullptr;
     auto error = aclGetRawTensorAddr(this->tensorPtr, &rawDataPtr);
     if (error != ACL_SUCCESS || !rawDataPtr) throw std::runtime_error(fmt::format("Failed to get tensor data pointer. error: {}", error));
     
     py::array result(this->dtype, this->shape);
     py::buffer_info info = result.request();
     if(tensorByteSize == 0) return result;
     
     if (this->aclDtype == ACL_FLOAT16 || this->aclDtype == ACL_BF16) {
         std::vector<uint16_t> temp_buffer(this->tensorSize);
         error = aclrtMemcpy(temp_buffer.data(), tensorByteSize, rawDataPtr, tensorByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
         if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("Failed to copy tensor data to host. error: {}", error));
         
         float* result_ptr = static_cast<float*>(info.ptr);
         for (size_t i = 0; i < this->tensorSize; ++i) {
             if (this->aclDtype == ACL_FLOAT16) {
                 uint16_t h = temp_buffer[i];
                 uint32_t f = ((h & 0x8000) << 16) | (((h & 0x7c00) + 0x1c000) << 13) | ((h & 0x03ff) << 13);
                 result_ptr[i] = *reinterpret_cast<float*>(&f);
             } else { // ACL_BF16
                 uint16_t bf = temp_buffer[i];
                 uint32_t f = (bf << 16);
                 result_ptr[i] = *reinterpret_cast<float*>(&f);
             }
         }
     } else {
         if(info.size * info.itemsize != tensorByteSize) throw std::runtime_error("Size mismatch between tensor and NumPy array");
         error = aclrtMemcpy(info.ptr, tensorByteSize, rawDataPtr, tensorByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
         if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("Failed to copy tensor data to host. error: {}", error));
     }
     
     return result;
 }
 
 
 /**
  * @brief Helper function to calculate total size of array.
  */
 int64_t NPUArray::GetShapeSize(const std::vector<int64_t>& shape) {
     int64_t shapeSize = 1;
     for(auto i : shape) {
         if(i <= 0) {
             throw std::runtime_error("Shape Dimensions Must Be Positive!");
         }
         shapeSize *= i;
     }
     return shapeSize;
 }
 
 
 /**
  * @brief Helper function to convert py::dtype to aclDataType.
  */
 aclDataType NPUArray::GetACLDataType(py::dtype dtype) {
     if(dtype.is(py::dtype::of<float>())) return ACL_FLOAT;
     if(dtype.is(py::dtype::of<double>())) return ACL_DOUBLE;
     if(dtype.is(py::dtype::of<int8_t>())) return ACL_INT8;
     if(dtype.is(py::dtype::of<int16_t>())) return ACL_INT16;
     if(dtype.is(py::dtype::of<int32_t>())) return ACL_INT32;
     if(dtype.is(py::dtype::of<int64_t>())) return ACL_INT64;
     if(dtype.is(py::dtype::of<uint8_t>())) return ACL_UINT8;
     if(dtype.is(py::dtype::of<uint16_t>())) return ACL_UINT16;
     if(dtype.is(py::dtype::of<uint32_t>())) return ACL_UINT32;
     if(dtype.is(py::dtype::of<uint64_t>())) return ACL_UINT64;
     if(dtype.is(py::dtype::of<bool>())) return ACL_BOOL;
     if(dtype.is(py::dtype::of<std::complex<float>>())) return ACL_COMPLEX64;
     if(dtype.is(py::dtype::of<std::complex<double>>())) return ACL_COMPLEX128;
     throw std::runtime_error("Unsupported py::dtype for aclDataType.");
 }
 
 
 /**
  * @brief Helper function to convert aclDataType to py::dtype.
  */
 py::dtype NPUArray::GetPyDtype(aclDataType acl_type) {
     switch (acl_type) {
         case ACL_FLOAT: return py::dtype::of<float>();
         case ACL_DOUBLE: return py::dtype::of<double>();
         case ACL_INT8: return py::dtype::of<int8_t>();
         case ACL_INT16: return py::dtype::of<int16_t>();
         case ACL_INT32: return py::dtype::of<int32_t>();
         case ACL_INT64: return py::dtype::of<int64_t>();
         case ACL_UINT8: return py::dtype::of<uint8_t>();
         case ACL_UINT16: return py::dtype::of<uint16_t>();
         case ACL_UINT32: return py::dtype::of<uint32_t>();
         case ACL_UINT64: return py::dtype::of<uint64_t>();
         case ACL_BOOL: return py::dtype::of<bool>();
         case ACL_FLOAT16: return py::dtype::of<float>();
         case ACL_BF16: return py::dtype::of<float>();
         case ACL_INT4: return py::dtype::of<uint8_t>();
         case ACL_UINT1: return py::dtype::of<uint8_t>();
         case ACL_COMPLEX64: return py::dtype::of<std::complex<float>>();
         case ACL_COMPLEX128: return py::dtype::of<std::complex<double>>();
         case ACL_COMPLEX32: return py::dtype::of<std::complex<float>>();
         case ACL_STRING: return py::dtype::of<char*>();
         case ACL_DT_UNDEFINED: return py::dtype::of<uint8_t>();
         case ACL_HIFLOAT8: return py::dtype::of<uint8_t>();
         case ACL_FLOAT8_E5M2: return py::dtype::of<uint8_t>();
         case ACL_FLOAT8_E4M3FN: return py::dtype::of<uint8_t>();
         case ACL_FLOAT8_E8M0: return py::dtype::of<uint8_t>();
         case ACL_FLOAT6_E3M2: return py::dtype::of<uint8_t>();
         case ACL_FLOAT6_E2M3: return py::dtype::of<uint8_t>();
         case ACL_FLOAT4_E2M1: return py::dtype::of<uint8_t>();
         case ACL_FLOAT4_E1M2: return py::dtype::of<uint8_t>();
         default:
             throw std::runtime_error("Unsupported aclDataType for py::dtype conversion.");
     }
 }
 
 
 /**
  * @brief Helper function to get byte size corresponding to aclDataType.
  */
 int64_t NPUArray::GetDataTypeSize(aclDataType dataType) {
     switch (dataType) {
         case ACL_FLOAT: return sizeof(float);
         case ACL_DOUBLE: return sizeof(double);
         case ACL_INT8: return sizeof(int8_t);
         case ACL_INT16: return sizeof(int16_t);
         case ACL_INT32: return sizeof(int32_t);
         case ACL_INT64: return sizeof(int64_t);
         case ACL_UINT8: return sizeof(uint8_t);
         case ACL_UINT16: return sizeof(uint16_t);
         case ACL_UINT32: return sizeof(uint32_t);
         case ACL_UINT64: return sizeof(uint64_t);
         case ACL_BOOL: return sizeof(bool);
         case ACL_FLOAT16: return sizeof(uint16_t);
         case ACL_BF16: return sizeof(uint16_t);
         case ACL_INT4: return 1;
         case ACL_UINT1: return 1;
         case ACL_COMPLEX64: return sizeof(std::complex<float>);
         case ACL_COMPLEX128: return sizeof(std::complex<double>);
         case ACL_COMPLEX32: return sizeof(std::complex<float>);
         case ACL_STRING: return sizeof(char*);
         case ACL_DT_UNDEFINED: return 0;
         case ACL_HIFLOAT8: return 1;
         case ACL_FLOAT8_E5M2: return 1;
         case ACL_FLOAT8_E4M3FN: return 1;
         case ACL_FLOAT8_E8M0: return 1;
         case ACL_FLOAT6_E3M2: return 1;
         case ACL_FLOAT6_E2M3: return 1;
         case ACL_FLOAT4_E2M1: return 1;
         case ACL_FLOAT4_E1M2: return 1;
         default:
             throw std::runtime_error("Unsupported aclDataType for size calculation.");
     }
 }
 
 
 std::vector<int64_t> GetBroadcastShape(const NPUArray& a, const NPUArray& b) {
     const std::vector<int64_t>& shapeA = a.shape;
     const std::vector<int64_t>& shapeB = b.shape;
 
     size_t ndimA = shapeA.size();
     size_t ndimB = shapeB.size();
     size_t ndimOut = std::max(ndimA, ndimB);
 
     std::vector<int64_t> result(ndimOut, 1);
 
     for (size_t i = 0; i < ndimOut; ++i) {
         int64_t dimA = (i < ndimA) ? shapeA[ndimA - 1 - i] : 1;
         int64_t dimB = (i < ndimB) ? shapeB[ndimB - 1 - i] : 1;
 
         if (dimA == dimB || dimA == 1 || dimB == 1) {
             result[ndimOut - 1 - i] = std::max(dimA, dimB);
         } else {
             throw std::invalid_argument(
                 "GetBroadcastShape: shapes are not broadcastable. "
                 "dimA=" + std::to_string(dimA) +
                 " dimB=" + std::to_string(dimB) +
                 " at axis -" + std::to_string(i + 1)
             );
         }
     }
 
     return result;
 }