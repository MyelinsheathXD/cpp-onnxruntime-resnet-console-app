#include <onnxruntime_cxx_api.h>
#include <iostream>

#include "Helpers.cpp"

void main()
{
	Ort::Env env;
	Ort::RunOptions runOptions;
	Ort::Session session(nullptr);

	constexpr int64_t numChannels = 3;
	constexpr int64_t width = 224;
	constexpr int64_t height = 224;
	constexpr int64_t numClasses = 1000;
	constexpr int64_t numInputElements = numChannels * height * width;


	const std::string imageFile = "E:\\AdaptiveCELLS\\VSprojects\\cpp\\pr\\2024\\01\\onnxcpp\\OnnxRuntimeResNet\\assets\\dog.png";
	const std::string labelFile = "E:\\AdaptiveCELLS\\VSprojects\\cpp\\pr\\2024\\01\\onnxcpp\\OnnxRuntimeResNet\\assets\\imagenet_classes.txt";
	auto modelPath = L"E:\\AdaptiveCELLS\\VSprojects\\cpp\\pr\\2024\\01\\onnxcpp\\OnnxRuntimeResNet\\assets\\resnet50v2.onnx";
	//auto modelPath = L"model_weights35.onnx";

	// Use CPU--------------------------------------------
	session = Ort::Session(env, modelPath, Ort::SessionOptions{ nullptr });

	// define shape
	const std::array<int64_t, 4> inputShape = { 1, numChannels, height, width };
	const std::array<int64_t, 2> outputShape = { 1, numClasses };

	// define array
	std::array<float, numInputElements> input;
	std::array<float, numClasses> results;

	// define Tensor
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
	auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

	// copy image data to input array
	//std::copy(imageVec.begin(), imageVec.end(), input.begin());



	// define names
	Ort::AllocatorWithDefaultOptions ort_alloc;
	Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
	Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
	const std::array<const char*, 1> inputNames = { inputName.get() };
	const std::array<const char*, 1> outputNames = { outputName.get() };
	inputName.release();
	outputName.release();


	// run inference-------------------------------------------------------
	try {
		session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
	}
	catch (Ort::Exception& e) {
		std::cout << e.what() << std::endl;
	}

	// sort results
	std::vector<std::pair<size_t, float>> indexValuePairs;
	for (size_t i = 0; i < results.size(); ++i) {
		indexValuePairs.emplace_back(i, results[i]);
	}
	std::sort(indexValuePairs.begin(), indexValuePairs.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

	// show Top5
	/*for (size_t i = 0; i < 5; ++i) {
		const auto& result = indexValuePairs[i];
		std::cout << i + 1 << ": " << labels[result.first] << " " << result.second << std::endl;
	}*/


}