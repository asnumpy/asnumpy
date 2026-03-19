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
#include <cstddef>


/**
 * @brief Constructor that creates an NPUArray with specified shape and data type.
 * 
 * Creates an array (aclTensor) stored on NPU by calling aclCreateTensor,
 * and initializes its shape and data type.
 * 
 * @param shape Vector containing array dimensions, defining the array shape.
 * @param dtype np.dtype defining the data type of array elements.
 * @throws std::runtime_error If memory allocation fails or data type is not supported.
 */
NPUArray::NPUArray(const std::vector<int64_t>& shape, py::dtype dtype) {
    // fmt::println("构造函数");
    this->shape = shape;
    this->dtype = dtype;
    this->aclDtype = GetACLDataType(dtype);
    tensorSize = GetShapeSize(shape);
    auto tensorByteSize = this->tensorSize * GetDataTypeSize(this->aclDtype);
    this->devicePtr = nullptr;
    auto error = aclrtMalloc(&this->devicePtr, tensorByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if(error != ACL_SUCCESS) {
        std::cout << "error = " << error << std::endl;
        throw std::runtime_error("NPUArray malloc error!");
    }
    this->strides.resize(this->shape.size());
    auto currentStride = 1;
    for(int64_t i = this->shape.size() - 1; i >= 0; i--) {
        this->strides[i] = currentStride;
        currentStride *= this->shape[i];
    }
    tensorPtr = aclCreateTensor(this->shape.data(), this->shape.size(), GetACLDataType(this->dtype), this->strides.data(), 0, ACL_FORMAT_ND, this->shape.data(), this->shape.size(), this->devicePtr);
}


/**
 * @brief Constructor that creates an NPUArray with specified shape and ACL data type.
 * 
 * Creates an array (aclTensor) stored on NPU by calling aclCreateTensor,
 * and initializes its shape and ACL data type directly.
 * This constructor bypasses NumPy dtype conversion for better performance.
 * 
 * @param shape Vector containing array dimensions, defining the array shape.
 * @param acl_type ACL data type constant.
 * @throws std::runtime_error If memory allocation fails or data type is not supported.
 */
NPUArray::NPUArray(const std::vector<int64_t>& shape, aclDataType acl_type) {
    this->shape = shape;
    this->aclDtype = acl_type;
    this->tensorSize = GetShapeSize(shape);
    auto tensorByteSize = this->tensorSize * GetDataTypeSize(this->aclDtype);
    
    // 直接使用 ACL 类型，不创建 NumPy dtype
    // 为了兼容性，创建一个空的 py::dtype 对象
    this->dtype = GetPyDtype(acl_type);
    
    void *devicePtr = nullptr;
    auto error = aclrtMalloc(&devicePtr, tensorByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if(error != ACL_SUCCESS) {
        std::cout << "error = " << error << std::endl;
        throw std::runtime_error("NPUArray malloc error!");
    }
    this->strides.resize(this->shape.size());
    auto currentStride = 1;
    for(int64_t i = this->shape.size() - 1; i >= 0; i--) {
        this->strides[i] = currentStride;
        currentStride *= this->shape[i];
    }
    tensorPtr = aclCreateTensor(this->shape.data(), this->shape.size(), acl_type, this->strides.data(), 0, ACL_FORMAT_ND, this->shape.data(), this->shape.size(), devicePtr);
}


/**
 * @brief Copy constructor - deep copy.
 * 
 * Creates a new NPUArray with the same content as the given NPUArray.
 * The new object owns its own memory space and is completely independent of the original.
 * 
 * @param other The NPUArray to copy from.
 */
NPUArray::NPUArray(const NPUArray& other) {
    // fmt::println("拷贝构造函数");
    this->shape = other.shape;
    this->dtype = other.dtype;
    this->aclDtype = other.aclDtype;
    this->tensorSize = other.tensorSize;
    this->strides = other.strides;
    auto tensorByteSize = this->tensorSize * GetDataTypeSize(this->aclDtype);
    void *devicePtr = nullptr;
    auto error = aclrtMalloc(&devicePtr, tensorByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if(error != ACL_SUCCESS) {
        throw std::runtime_error("NPUArray copy constructor malloc error!");
    }
    this->tensorPtr = aclCreateTensor(this->shape.data(), this->shape.size(), this->aclDtype, this->strides.data(), 0, ACL_FORMAT_ND, this->shape.data(), this->shape.size(), devicePtr);
    void* srcPtr = nullptr;
    error = aclGetRawTensorAddr(other.tensorPtr, &srcPtr);
    if(error != ACL_SUCCESS || !srcPtr) throw std::runtime_error(fmt::format("Failed to get source tensor data pointer. error: {}", error));
    error = aclrtMemcpy(devicePtr, tensorByteSize, srcPtr, tensorByteSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("Failed to copy tensor data. error: {}", error));
    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("Failed to synchronize after tensor copy. error: {}", error));
}


/**
 * @brief Move constructor.
 * 
 * Transfers ownership of resources from the given NPUArray to the new object.
 * The original object is left in an invalid state.
 * 
 * @param other The NPUArray to move from.
 */
NPUArray::NPUArray(NPUArray&& other) noexcept {
    // fmt::println("移动构造函数");
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
 * 
 * Implements deep copy assignment, ensuring the current object is completely independent of the right-hand object.
 * 
 * @param other The NPUArray to copy from.
 * @return Reference to this NPUArray.
 */
NPUArray& NPUArray::operator=(const NPUArray& other) {
    // fmt::println("拷贝赋值运算符");
    if(this != &other) {
        if(this->tensorPtr) {
            aclDestroyTensor(this->tensorPtr);
        }
        this->shape = other.shape;
        this->dtype = other.dtype;
        this->aclDtype = other.aclDtype;
        this->tensorSize = other.tensorSize;
        this->strides = other.strides;


        auto tensorByteSize = this->tensorSize * GetDataTypeSize(this->aclDtype);
        void *devicePtr = nullptr;
        auto error = aclrtMalloc(&devicePtr, tensorByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) {
            throw std::runtime_error("NPUArray copy assignment malloc error!");
        }
        this->tensorPtr = aclCreateTensor(this->shape.data(), this->shape.size(), this->aclDtype, this->strides.data(), 0, ACL_FORMAT_ND, this->shape.data(), this->shape.size(), devicePtr);
        void* srcPtr = nullptr;
        error = aclGetRawTensorAddr(other.tensorPtr, &srcPtr);
        if(error != ACL_SUCCESS || !srcPtr) throw std::runtime_error(fmt::format("Failed to get source tensor data pointer. error: {}", error));
        error = aclrtMemcpy(devicePtr, tensorByteSize, srcPtr, tensorByteSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("Failed to copy tensor data. error: {}", error));
        error = aclrtSynchronizeDevice();
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("Failed to synchronize after tensor copy. error: {}", error));
    }
    return *this;
}


/**
 * @brief Move assignment operator.
 * 
 * Transfers ownership of resources from the given NPUArray to the current object.
 * The original object is left in an invalid state.
 * 
 * @param other The NPUArray to move from.
 * @return Reference to this NPUArray.
 */
NPUArray& NPUArray::operator=(NPUArray&& other) noexcept {
    // fmt::println("移动赋值运算符");
    if(this != &other) {
        if(this->tensorPtr) {
            aclDestroyTensor(this->tensorPtr);
        }
        this->tensorPtr = other.tensorPtr;
        this->shape = std::move(other.shape);
        this->dtype = other.dtype;
        this->aclDtype = other.aclDtype;
        this->tensorSize = other.tensorSize;
        this->strides = std::move(other.strides);
        other.tensorPtr = nullptr;
    }
    return *this;
}


/**
 * @brief Destructor that releases resources occupied by NPUArray.
 */
NPUArray::~NPUArray() {
    // fmt::println("析构函数");
    if(this->tensorPtr) {
        // fmt::println("析构函数：销毁aclTensor");
        auto error = aclDestroyTensor(this->tensorPtr);
        this->tensorPtr = nullptr;
    }
    if (this->devicePtr) {
        // fmt::println("析构函数：aclrtFree");
        auto error = aclrtFree(this->devicePtr);
        this->devicePtr = nullptr;
    }
}


/**
 * @brief Static method to create NPUArray from NumPy array.
 * 
 * Creates an NPUArray from a NumPy array and copies data
 * from host memory to NPU device memory.
 * 
 * @param host_data Input NumPy array.
 * @return NPUArray The created NPUArray.
 * @throws std::runtime_error If getting tensor data pointer fails or data copy fails.
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
 * 
 * Copies data from NPU device memory to host memory and returns a NumPy array.
 * 
 * @return py::array The converted NumPy array.
 * @throws std::runtime_error If getting tensor data pointer fails or data copy fails.
 * @throws std::runtime_error If tensor size doesn't match NumPy array size.
 */
py::array NPUArray::ToNumpy() const {
    auto tensorByteSize = this->tensorSize * GetDataTypeSize(this->aclDtype);
    void* rawDataPtr = nullptr;
    auto error = aclGetRawTensorAddr(this->tensorPtr, &rawDataPtr);
    if (error != ACL_SUCCESS || !rawDataPtr) throw std::runtime_error(fmt::format("Failed to get tensor data pointer. error: {}", error));
    
    // 创建结果数组
    py::array result(this->dtype, this->shape);
    py::buffer_info info = result.request();
    if(tensorByteSize == 0) return result;
    
    // 对于特殊类型，需要特殊处理
    if (this->aclDtype == ACL_FLOAT16 || this->aclDtype == ACL_BF16) {
        // 对于 float16 和 bf16，我们需要先复制到临时缓冲区，然后转换
        std::vector<uint16_t> temp_buffer(this->tensorSize);
        error = aclrtMemcpy(temp_buffer.data(), tensorByteSize, rawDataPtr, tensorByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("Failed to copy tensor data to host. error: {}", error));
        
        // 转换为 float32
        float* result_ptr = static_cast<float*>(info.ptr);
        for (size_t i = 0; i < this->tensorSize; ++i) {
            if (this->aclDtype == ACL_FLOAT16) {
                // 简单的 float16 到 float32 转换
                uint16_t h = temp_buffer[i];
                uint32_t f = ((h & 0x8000) << 16) | (((h & 0x7c00) + 0x1c000) << 13) | ((h & 0x03ff) << 13);
                result_ptr[i] = *reinterpret_cast<float*>(&f);
            } else { // ACL_BF16
                // bfloat16 到 float32 转换
                uint16_t bf = temp_buffer[i];
                uint32_t f = (bf << 16);
                result_ptr[i] = *reinterpret_cast<float*>(&f);
            }
        }
    } else {
        // 对于其他类型，直接复制
        if(info.size * info.itemsize != tensorByteSize) throw std::runtime_error("Size mismatch between tensor and NumPy array");
        error = aclrtMemcpy(info.ptr, tensorByteSize, rawDataPtr, tensorByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("Failed to copy tensor data to host. error: {}", error));
    }
    
    return result;
}


/**
 * @brief Helper function to calculate total size of array.
 * 
 * Calculates the total number of elements for a given shape.
 * 
 * @param shape Vector containing array dimensions, defining the array shape.
 * @return int64_t Total number of elements in the array.
 * @throws std::runtime_error If any dimension in shape is less than or equal to 0.
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
 * 
 * Converts Python data type to ACL data type for NPU operations.
 * 
 * @param dtype Input py::dtype.
 * @return aclDataType The converted aclDataType.
 * @throws std::runtime_error If input data type is not supported.
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
 * 
 * Converts ACL data type to Python data type for NumPy compatibility.
 * 
 * @param acl_type Input aclDataType.
 * @return py::dtype The converted py::dtype.
 * @throws std::runtime_error If input data type is not supported.
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
        case ACL_FLOAT16: return py::dtype::of<float>();  // float16 映射到 float，保持浮点语义
        case ACL_BF16: return py::dtype::of<float>();     // bf16 映射到 float，保持浮点语义
        case ACL_INT4: return py::dtype::of<uint8_t>();      // int4 映射到 uint8
        case ACL_UINT1: return py::dtype::of<uint8_t>();     // uint1 映射到 uint8
        case ACL_COMPLEX64: return py::dtype::of<std::complex<float>>();
        case ACL_COMPLEX128: return py::dtype::of<std::complex<double>>();
        case ACL_COMPLEX32: return py::dtype::of<std::complex<float>>(); // complex32 映射到 complex64
        case ACL_STRING: return py::dtype::of<char*>();      // 字符串指针
        case ACL_DT_UNDEFINED: return py::dtype::of<uint8_t>(); // 未定义类型映射到 uint8
        case ACL_HIFLOAT8: return py::dtype::of<uint8_t>();  // Float8 变体映射到 uint8
        case ACL_FLOAT8_E5M2: return py::dtype::of<uint8_t>(); // Float8 E5M2格式映射到 uint8
        case ACL_FLOAT8_E4M3FN: return py::dtype::of<uint8_t>(); // Float8 E4M3FN格式映射到 uint8
        case ACL_FLOAT8_E8M0: return py::dtype::of<uint8_t>(); // Float8 E8M0格式映射到 uint8
        case ACL_FLOAT6_E3M2: return py::dtype::of<uint8_t>(); // Float6 E3M2格式映射到 uint8
        case ACL_FLOAT6_E2M3: return py::dtype::of<uint8_t>(); // Float6 E2M3格式映射到 uint8
        case ACL_FLOAT4_E2M1: return py::dtype::of<uint8_t>(); // Float4 E2M1格式映射到 uint8
        case ACL_FLOAT4_E1M2: return py::dtype::of<uint8_t>(); // Float4 E1M2格式映射到 uint8
        default:
            throw std::runtime_error("Unsupported aclDataType for py::dtype conversion.");
    }
}


/**
 * @brief Helper function to get byte size corresponding to aclDataType.
 * 
 * Returns the byte size corresponding to ACL data type for memory allocation.
 * 
 * @param dataType Input aclDataType.
 * @return int64_t Byte size of the data type.
 * @throws std::runtime_error If input data type is not supported.
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
        case ACL_FLOAT16: return sizeof(uint16_t);  // 2字节
        case ACL_BF16: return sizeof(uint16_t);     // 2字节
        case ACL_INT4: return 1;                    // 4位，但按字节对齐
        case ACL_UINT1: return 1;                   // 1位，但按字节对齐
        case ACL_COMPLEX64: return sizeof(std::complex<float>);
        case ACL_COMPLEX128: return sizeof(std::complex<double>);
        case ACL_COMPLEX32: return sizeof(std::complex<float>); // complex32 maps to complex64
        case ACL_STRING: return sizeof(char*);      // 字符串指针
        case ACL_DT_UNDEFINED: return 0;            // 未定义类型
        case ACL_HIFLOAT8: return 1;                // Float8 变体，1字节
        case ACL_FLOAT8_E5M2: return 1;             // Float8 E5M2格式，1字节
        case ACL_FLOAT8_E4M3FN: return 1;           // Float8 E4M3FN格式，1字节
        case ACL_FLOAT8_E8M0: return 1;             // Float8 E8M0格式，1字节
        case ACL_FLOAT6_E3M2: return 1;             // Float6 E3M2格式，1字节
        case ACL_FLOAT6_E2M3: return 1;             // Float6 E2M3格式，1字节
        case ACL_FLOAT4_E2M1: return 1;             // Float4 E2M1格式，1字节
        case ACL_FLOAT4_E1M2: return 1;             // Float4 E1M2格式，1字节
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

