#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp> 

#include <cuda_runtime.h>  // cuda
#include <cuda_runtime_api.h>

#include "NvInfer.h"  //TensorRT 
#include "NvInferRuntime.h"

#include "buf.h"
#include "engine.h"
#include "logging.h"


using namespace std;
using namespace cv;


int main()
{
	OnnxParams params;
	params.onnxFilePath = "seg.onnx";
	params.engineFilePath = "seg.engine";
	params.channel = 1;
	params.fp16 = false;
	TrtEngine tEngine;

	fstream file;
	file.open(params.engineFilePath, ios::binary | ios::in);
	// 不存在则生成并保存引擎文件
	if (!file.is_open()) {
		tEngine.buildEngine(params);
	}
	else {
		tEngine.loadEngine(file);
	}

	nvinfer1::IExecutionContext* context = tEngine.mEngine->createExecutionContext();
	Mat img = imread("depvd.bmp", IMREAD_UNCHANGED);
	int height = img.rows;
	int width = img.cols;
	//resize(img, img, cv::Size(256, 256), 0, 0, INTER_LINEAR); //resize到任意大小都可
	img.convertTo(img, CV_32F, 1.0 / 255);

	nvinfer1::Dims inDims{ 4, 1, 1,height, width };
	nvinfer1::Dims outDims{ 3, 1, height, width };

	// BufferManager 调用方式
	{
		samplesCommon::BufferManager buffers(tEngine.mEngine, inDims, outDims);
		float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer("input"));
		memcpy_s(hostDataBuffer, height * width * sizeof(float), img.data, height * width * sizeof(float));
		buffers.copyInputToDevice();

		context->setBindingDimensions(0, inDims);
		bool status = context->executeV2(buffers.getDeviceBindings().data());

		float* output = static_cast<float*>(buffers.getHostBuffer("output"));
		buffers.copyOutputToHost();

		cv::Mat res = cv::Mat(height, width, CV_32F, output).clone();
		imshow("img", img);
		imshow("res1", res * 255);
		//waitKey(0);
	}

	// 自写数据方式调用
	{
		float* input_layer_;
		size_t input_layer_size = samplesCommon::volume(inDims) * sizeof(float);
		cudaError_t st = cudaMalloc(&input_layer_, input_layer_size);
		float* cpu_input_layer_ = new float[input_layer_size];

		memcpy_s(cpu_input_layer_, height * width * sizeof(float), img.data, height * width * sizeof(float));
		cudaMemcpy(input_layer_, cpu_input_layer_, input_layer_size, cudaMemcpyHostToDevice);
		delete[]cpu_input_layer_;

		size_t output_layer_size = samplesCommon::volume(outDims) * sizeof(float);
		float* output_layer_;
		st = cudaMalloc(&output_layer_, output_layer_size);
		void* buffers_[2] = { input_layer_, output_layer_ };

		context->setBindingDimensions(0, inDims);
		bool status = context->executeV2(buffers_);
		float* cpu_output_layer_ = new float[output_layer_size];
		cudaMemcpy(cpu_output_layer_, output_layer_, output_layer_size, cudaMemcpyDeviceToHost);
		cv::Mat res = cv::Mat(height, width, CV_32F, cpu_output_layer_).clone();
		imshow("res2", res*255);
		waitKey(0);
		delete[]cpu_output_layer_;
		cudaFree(input_layer_);
		cudaFree(output_layer_);
	}

}

