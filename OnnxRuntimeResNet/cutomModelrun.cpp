#include <onnxruntime_cxx_api.h>
#include <iostream>
// C++ program to generate random float numbers
//#include <bits/stdc++.h>

using namespace std;


#include "Helpers.cpp"


float randomFloat()
{
	return (float)(rand()) / (float)(RAND_MAX);
}

void main()
{
	Ort::Env env;
	Ort::RunOptions runOptions;
	Ort::Session session(nullptr);


	constexpr int64_t numOutputElements = 30;
	constexpr int64_t numInputElements = 35;

	auto modelPath = L"E:\\AdaptiveCELLS\\VSprojects\\cpp\\gits\\2024\\01\\cpp-onnxruntime-resnet-console-app\\OnnxRuntimeResNet\\assets\\ImageClassifier.onnx";
	

	// Use CPU--------------------------------------------
	session = Ort::Session(env, modelPath, Ort::SessionOptions{ nullptr });

	// define shape
	const std::array<int64_t, 2> inputShape = { 1,35 };
	const std::array<int64_t, 2> outputShape = { 1, 30 };

	// define array
	std::array<float, numInputElements> input;
	std::array<float, numOutputElements> results;

	for (int i = 0; i < input.size(); i++) {
		input[i] = randomFloat();
	}

	// define Tensor
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
	auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());




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



	// Get pointer to output tensor float values
	float* floatarr = inputTensor.GetTensorMutableData<float>();
	//int* floatarr = outputTensor[0].GetTensorMutableData<int>();
	//std::cout<< floatarr[0] << std::endl;
	for (size_t i = 0; i < numInputElements; i++)
	{
		std::cout << i + 1 << ": " << floatarr[i] << " " <<"input " << std::endl;
	}

	// Get pointer to output tensor float values
	float* floatarr2 = outputTensor.GetTensorMutableData<float>();
	//int* floatarr = outputTensor[0].GetTensorMutableData<int>();
	//std::cout << floatarr[0] << std::endl;
	for (size_t i = 0; i < numOutputElements; i++)
	{
		std::cout << i + 1 << ": " << floatarr2[i] << " " << "output " << std::endl;
	}

}