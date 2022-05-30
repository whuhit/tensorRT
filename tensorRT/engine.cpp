#include <iostream>
#include <fstream>
#include <sstream>

#include "logging.h"
#include "engine.h"

void print_() { std::cout << std::endl; }
//
template<typename T, typename... Ts>
void print_(T arg1, Ts... arg_left) {
	std::cout << arg1 << " ";
	print_(arg_left...);
}

int TrtEngine::buildEngine(const OnnxParams params)
{
	print_("build onnx:", params.onnxFilePath);
	print_("this will take few minutes, please wait");
	auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(mLogger));
	if (!builder)
	{
		cout << "createInferBuilder error" << endl;
		return -1;
	}
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
	if (!network)
	{
		cout << "createNetworkV2 error" << endl;
		return -1;
	}
	auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!config)
	{
		cout << "createBuilderConfig error" << endl;
		return -1;
	}
	config->setMaxWorkspaceSize(2048_MiB);
	if (params.fp16)
	{
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}

	// 设置profile 这里有个OptProfileSelector，这个用来设置优化的参数,比如（Tensor的形状或者动态尺寸），
	nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
	profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(params.min_batch, params.channel, params.min_height, params.min_width));
	profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(params.opt_batch, params.channel, params.opt_height, params.opt_width));
	profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(params.max_batch, params.channel, params.max_height, params.max_width));
	config->addOptimizationProfile(profile);

	auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, mLogger));
	if (!parser)
	{
		cout << "createParser error" << endl;
		return -1;
	}

	auto parsed = parser->parseFromFile(params.onnxFilePath.c_str(), 3);
	if (!parsed)
	{
		cout << "parse onnx File error" << endl;
		return -1;
	}

	auto profileStream = makeCudaStream();
	if (!profileStream)
	{
		cout << "makeCudaStream error";
		return -1;
	}
	config->setProfileStream(*profileStream);

	SampleUniquePtr<nvinfer1::IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
	if (!plan)
	{
		cout << "builder->buildSerializedNetwork error" << endl;
		return -1;
	}

	SampleUniquePtr<nvinfer1::IRuntime> runtime{ nvinfer1::createInferRuntime(mLogger) };
	if (!runtime)
	{
		cout << "createInferRuntime error" << endl;
		return -1;
	}

	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
		runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
	if (!mEngine)
	{
		cout << "deserializeCudaEngine error" << endl;
		return -1;
	}
	//save
	SampleUniquePtr<nvinfer1::IHostMemory> serializedModel(mEngine->serialize());
	std::ofstream p(params.engineFilePath.c_str(), std::ios::binary);
	p.write((const char*)serializedModel->data(), serializedModel->size());
	p.close();
	return 0;
}

int TrtEngine::loadEngine(fstream& file)
{
	file.seekg(0, std::ios::end);
	int length = file.tellg();
	file.seekg(0, std::ios::beg);
	std::unique_ptr<char[]> data(new char[length]);
	file.read(data.get(), length);
	file.close();
	SampleUniquePtr<nvinfer1::IRuntime> runTime(nvinfer1::createInferRuntime(mLogger));
	if (runTime == nullptr) {
		return -1;
	}
	nvinfer1::ICudaEngine* engine = runTime->deserializeCudaEngine(data.get(), length, nullptr);
	if (engine == nullptr) {
		return -1;
	}
	mEngine = shared_ptr<nvinfer1::ICudaEngine>(engine, InferDeleter());
	return 0;
}
