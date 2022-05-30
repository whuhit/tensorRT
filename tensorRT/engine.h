#pragma once
#include <string>
//#include <cuda_runtime.h>  // cuda
#include <cuda_runtime_api.h>

#include "NvInferRuntime.h" //TensorRT include
#include "NvInfer.h"  
#include "NvOnnxParser.h"

#include "logging.h"

using namespace std;

struct InferDeleter
{
	template <typename T>
	void operator()(T* obj) const
	{
		delete obj;
	}
};

// \! ����ָ��, ��������TRT�����м����͵Ķ�ռ����ָ��
template <typename T>
using SampleUniquePtr = unique_ptr<T, InferDeleter>;


// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _MiB(unsigned long long val)
{
	return val * (1 << 20);
}

static auto StreamDeleter = [](cudaStream_t* pStream)
{
	if (pStream)
	{
		cudaStreamDestroy(*pStream);
		delete pStream;
	}
};
inline unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
{
	unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
	if (cudaStreamCreate(pStream.get()) != cudaSuccess)
	{
		pStream.reset(nullptr);
	}

	return pStream;
}

// \! �����ʼ�� ����
struct OnnxParams
{
	// ��ͬƽ̨�Ĺ�������
	string onnxFilePath;	                    // onnx�ļ�·��
	string engineFilePath;	                    // �����ļ�����λ��
	bool fp16{ false };	                        // �뾫�ȼ���
	int channel = 1;                            // ͨ��
	int max_batch = 16;
	int max_height = 512;
	int max_width = 512;
	int min_batch = 1;
	int min_height = 128;
	int min_width = 128;
	int opt_batch = 8;
	int opt_height = 256;
	int opt_width = 256;
};


class TrtEngine {
public:
	int buildEngine(const OnnxParams params);
	int loadEngine(fstream& file);  // ���湹��

	std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

private:
	sample::Logger mLogger = sample::Logger(nvinfer1::ILogger::Severity::kINFO);
};